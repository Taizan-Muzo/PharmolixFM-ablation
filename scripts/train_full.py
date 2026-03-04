"""
PharmolixFM 完整训练脚本
支持真实数据训练和虚拟数据训练
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM
from utils.config import Config
from data.dataset import DummyDataset
from data.crossdocked_dataset import CrossDockedDataset


def collate_fn(batch):
    """批次整理 - 拼接多个样本"""
    if len(batch) == 1:
        return batch[0]
    
    molecules_list, pockets_list, metadata_list = zip(*batch)
    
    cumsum_nodes = 0
    cumsum_pocket_nodes = 0
    
    batch_mol = {key: [] for key in ['node_type', 'pos', 'halfedge_type', 'halfedge_index', 'batch']}
    batch_pocket = {key: [] for key in ['atom_feature', 'pos', 'knn_edge_index', 'batch']}
    
    for i, (mol, pocket) in enumerate(zip(molecules_list, pockets_list)):
        num_nodes = mol['pos'].shape[0]
        num_pocket_nodes = pocket['pos'].shape[0]
        
        batch_mol['node_type'].append(mol['node_type'])
        batch_mol['pos'].append(mol['pos'])
        batch_mol['halfedge_type'].append(mol['halfedge_type'])
        
        if 'halfedge_index' in mol and mol['halfedge_index'].shape[1] > 0:
            batch_mol['halfedge_index'].append(mol['halfedge_index'] + cumsum_nodes)
        batch_mol['batch'].append(torch.full((num_nodes,), i, dtype=torch.long))
        
        batch_pocket['atom_feature'].append(pocket['atom_feature'])
        batch_pocket['pos'].append(pocket['pos'])
        
        if 'knn_edge_index' in pocket and pocket['knn_edge_index'].shape[1] > 0:
            batch_pocket['knn_edge_index'].append(pocket['knn_edge_index'] + cumsum_pocket_nodes)
        batch_pocket['batch'].append(torch.full((num_pocket_nodes,), i, dtype=torch.long))
        
        cumsum_nodes += num_nodes
        cumsum_pocket_nodes += num_pocket_nodes
    
    result_mol = {}
    for key in batch_mol:
        if batch_mol[key]:
            dim = 1 if 'index' in key else 0
            result_mol[key] = torch.cat(batch_mol[key], dim=dim)
    
    result_pocket = {}
    for key in batch_pocket:
        if batch_pocket[key]:
            dim = 1 if 'index' in key else 0
            result_pocket[key] = torch.cat(batch_pocket[key], dim=dim)
    
    result_mol['batch'] = torch.cat([torch.full((m['pos'].shape[0],), i, dtype=torch.long) 
                                      for i, m in enumerate(molecules_list)])
    result_pocket['batch'] = torch.cat([torch.full((p['pos'].shape[0],), i, dtype=torch.long) 
                                         for i, p in enumerate(pockets_list)])
    
    return result_mol, result_pocket, metadata_list


def train_epoch(model, dataloader, optimizer, device, epoch, log_interval=10):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_pos_loss = 0.0
    total_node_loss = 0.0
    total_edge_loss = 0.0
    num_batches = 0
    
    for batch_idx, (molecules, pockets, metadata) in enumerate(dataloader):
        # 移到设备
        for key in molecules:
            if isinstance(molecules[key], torch.Tensor):
                molecules[key] = molecules[key].to(device)
        for key in pockets:
            if isinstance(pockets[key], torch.Tensor):
                pockets[key] = pockets[key].to(device)
        
        optimizer.zero_grad()
        
        try:
            # 使用 batch 版本前向传播
            loss_dict = model.forward_pocket_molecule_docking_batch(pockets, molecules)
            loss = loss_dict['loss']
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_pos_loss += loss_dict.get('pos_loss', 0).item() if isinstance(loss_dict.get('pos_loss'), torch.Tensor) else 0
            total_node_loss += loss_dict.get('node_loss', 0).item() if isinstance(loss_dict.get('node_loss'), torch.Tensor) else 0
            total_edge_loss += loss_dict.get('edge_loss', 0).item() if isinstance(loss_dict.get('edge_loss'), torch.Tensor) else 0
            num_batches += 1
            
            if batch_idx % log_interval == 0:
                batch_size = molecules['batch'].max().item() + 1
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f} (batch_size={batch_size})")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'pos_loss': total_pos_loss / max(num_batches, 1),
        'node_loss': total_node_loss / max(num_batches, 1),
        'edge_loss': total_edge_loss / max(num_batches, 1),
    }


def validate(model, dataloader, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (molecules, pockets, metadata) in enumerate(dataloader):
            # 移到设备
            for key in molecules:
                if isinstance(molecules[key], torch.Tensor):
                    molecules[key] = molecules[key].to(device)
            for key in pockets:
                if isinstance(pockets[key], torch.Tensor):
                    pockets[key] = pockets[key].to(device)
            
            try:
                loss_dict = model.forward_pocket_molecule_docking_batch(pockets, molecules)
                loss = loss_dict['loss']
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train PharmolixFM")
    parser.add_argument("--data_dir", type=str, default="data/",
                       help="数据目录")
    parser.add_argument("--output_dir", type=str, default="checkpoints/",
                       help="输出目录")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="权重衰减")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="计算设备")
    parser.add_argument("--use_dummy", action="store_true",
                       help="使用虚拟数据")
    parser.add_argument("--save_every", type=int, default=10,
                       help="每多少 epoch 保存一次")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="日志间隔")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="数据加载 workers")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的检查点路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_dict = vars(args)
    config_dict['start_time'] = datetime.now().isoformat()
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("=" * 60)
    print("PharmolixFM 训练")
    print("=" * 60)
    print(f"配置: {json.dumps(config_dict, indent=2)}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    config = Config()
    model = PharmolixFM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {num_params:,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    start_epoch = 1
    best_val_loss = float('inf')
    
    # 恢复训练
    if args.resume:
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    
    # 创建数据集
    if args.use_dummy:
        print("使用虚拟数据集")
        train_dataset = DummyDataset(num_samples=1000)
        val_dataset = DummyDataset(num_samples=100)
    else:
        print(f"加载真实数据: {args.data_dir}")
        # 加载数据集并手动划分 train/val
        from data.crossdocked_dataset import CrossDockedDataset
        full_dataset = CrossDockedDataset(
            data_dir=args.data_dir, split="train", debug=False
        )
        
        # 手动划分: 90% train, 10% val
        dataset_size = len(full_dataset)
        val_size = min(int(dataset_size * 0.1), 100)  # 最多 100 个验证样本
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers
    )
    
    # 训练日志
    training_history = []
    
    print("\n开始训练...")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, args.log_interval
        )
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, "
              f"Pos: {train_metrics['pos_loss']:.4f}, "
              f"Node: {train_metrics['node_loss']:.4f}, "
              f"Edge: {train_metrics['edge_loss']:.4f}")
        
        # 验证
        val_loss = validate(model, val_loader, device)
        print(f"验证 - Loss: {val_loss:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录历史
        training_history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_pos_loss': train_metrics['pos_loss'],
            'train_node_loss': train_metrics['node_loss'],
            'train_edge_loss': train_metrics['edge_loss'],
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
        })
        
        # 保存检查点
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_metrics['loss'],
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"保存最佳模型 (val_loss: {val_loss:.4f})")
    
    # 保存最终模型
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    
    # 保存训练历史
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
