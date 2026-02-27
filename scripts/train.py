"""
PharmolixFM 训练脚本（支持任意 batch size）
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM
from utils.config import Config
from data.dataset import DummyDataset, PocketMoleculeDataset


def collate_fn(batch):
    """
    自定义批次整理函数 - 支持任意 batch size
    
    将多个样本拼接成一个 batch，处理 edge_index 的偏移
    """
    if len(batch) == 1:
        return batch[0]
    
    molecules_list, pockets_list, metadata_list = zip(*batch)
    
    # 合并分子数据
    batch_mol = {}
    batch_pocket = {}
    
    # 计算累积节点数（用于 edge_index 偏移）
    cumsum_nodes = 0
    cumsum_pocket_nodes = 0
    
    mol_keys = ['node_type', 'pos', 'halfedge_type', 'halfedge_index']
    pocket_keys = ['atom_feature', 'pos', 'knn_edge_index']
    
    # 初始化
    for key in mol_keys:
        batch_mol[key] = []
    for key in pocket_keys:
        batch_pocket[key] = []
    
    batch_mol['t_pos'] = []
    batch_mol['t_node_type'] = []
    batch_mol['t_halfedge_type'] = []
    batch_mol['fixed_halfdist'] = []
    
    for i, (mol, pocket) in enumerate(zip(molecules_list, pockets_list)):
        num_nodes = mol['pos'].shape[0]
        num_pocket_nodes = pocket['pos'].shape[0]
        
        # 节点特征和位置 - 直接拼接
        batch_mol['node_type'].append(mol['node_type'])
        batch_mol['pos'].append(mol['pos'])
        batch_mol['halfedge_type'].append(mol['halfedge_type'])
        
        # edge_index 需要加上偏移
        if 'halfedge_index' in mol and mol['halfedge_index'].shape[1] > 0:
            offset_edge_index = mol['halfedge_index'] + cumsum_nodes
            batch_mol['halfedge_index'].append(offset_edge_index)
        
        # 时间步和固定特征（每个样本一个值，扩展到所有节点）
        t = torch.tensor([0.5])  # 默认时间步，实际应该从数据获取
        batch_mol['t_pos'].append(t.expand(num_nodes))
        batch_mol['t_node_type'].append(t.expand(num_nodes))
        batch_mol['t_halfedge_type'].append(t.expand(mol['halfedge_type'].shape[0]))
        batch_mol['fixed_halfdist'].append(torch.zeros(mol['halfedge_type'].shape[0]))
        
        # 口袋数据
        batch_pocket['atom_feature'].append(pocket['atom_feature'])
        batch_pocket['pos'].append(pocket['pos'])
        
        if 'knn_edge_index' in pocket and pocket['knn_edge_index'].shape[1] > 0:
            offset_pocket_edge = pocket['knn_edge_index'] + cumsum_pocket_nodes
            batch_pocket['knn_edge_index'].append(offset_pocket_edge)
        
        cumsum_nodes += num_nodes
        cumsum_pocket_nodes += num_pocket_nodes
    
    # 拼接所有张量
    result_mol = {}
    for key in batch_mol:
        if batch_mol[key]:
            result_mol[key] = torch.cat(batch_mol[key], dim=0 if key != 'halfedge_index' and key != 'knn_edge_index' else 1)
    
    result_pocket = {}
    for key in batch_pocket:
        if batch_pocket[key]:
            result_pocket[key] = torch.cat(batch_pocket[key], dim=0 if key != 'knn_edge_index' else 1)
    
    # 添加 batch 向量（指示每个节点属于哪个样本）
    result_mol['batch'] = torch.cat([torch.full((m['pos'].shape[0],), i, dtype=torch.long) 
                                      for i, m in enumerate(molecules_list)])
    result_pocket['batch'] = torch.cat([torch.full((p['pos'].shape[0],), i, dtype=torch.long) 
                                         for i, p in enumerate(pockets_list)])
    
    return result_mol, result_pocket, metadata_list


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        # 处理不同 batch size 的情况
        if isinstance(batch_data, tuple) and len(batch_data) == 3:
            molecules, pockets, metadata = batch_data
        else:
            # batch_size=1 的情况
            molecules, pockets, metadata = batch_data[0], batch_data[1], batch_data[2]
        
        # 将数据移到设备
        for key in list(molecules.keys()):
            if isinstance(molecules[key], torch.Tensor):
                molecules[key] = molecules[key].to(device)
        for key in list(pockets.keys()):
            if isinstance(pockets[key], torch.Tensor):
                pockets[key] = pockets[key].to(device)
        
        optimizer.zero_grad()
        
        try:
            # 前向传播 - 需要处理 batch
            # 目前 model_forward 不支持 batch，需要逐个样本处理
            batch_size = molecules['batch'].max().item() + 1 if 'batch' in molecules else 1
            
            total_batch_loss = 0.0
            valid_samples = 0
            
            for i in range(batch_size):
                # 提取单个样本
                if 'batch' in molecules:
                    mask = molecules['batch'] == i
                    mol_sample = {k: v[mask] if k != 'halfedge_index' and k != 'batch' else v 
                                 for k, v in molecules.items()}
                    # 重新索引 edge_index
                    if 'halfedge_index' in mol_sample:
                        node_indices = torch.where(mask)[0]
                        idx_map = {int(old): new for new, old in enumerate(node_indices)}
                        old_edges = mol_sample['halfedge_index']
                        new_edges = torch.zeros_like(old_edges)
                        for e in range(old_edges.shape[1]):
                            new_edges[0, e] = idx_map.get(int(old_edges[0, e]), 0)
                            new_edges[1, e] = idx_map.get(int(old_edges[1, e]), 0)
                        mol_sample['halfedge_index'] = new_edges
                    
                    p_mask = pockets['batch'] == i
                    pocket_sample = {k: v[p_mask] if k != 'knn_edge_index' and k != 'batch' else v 
                                    for k, v in pockets.items()}
                    if 'knn_edge_index' in pocket_sample:
                        p_node_indices = torch.where(p_mask)[0]
                        p_idx_map = {int(old): new for new, old in enumerate(p_node_indices)}
                        old_p_edges = pocket_sample['knn_edge_index']
                        new_p_edges = torch.zeros_like(old_p_edges)
                        for e in range(old_p_edges.shape[1]):
                            new_p_edges[0, e] = p_idx_map.get(int(old_p_edges[0, e]), 0)
                            new_p_edges[1, e] = p_idx_map.get(int(old_p_edges[1, e]), 0)
                        pocket_sample['knn_edge_index'] = new_p_edges
                else:
                    mol_sample = molecules
                    pocket_sample = pockets
                
                # 跳过空样本
                if mol_sample['pos'].shape[0] == 0:
                    continue
                
                # 前向传播
                loss_dict = model.forward_pocket_molecule_docking(pocket_sample, mol_sample)
                loss = loss_dict['loss']
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    total_batch_loss += loss.item()
                    valid_samples += 1
            
            if valid_samples > 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                avg_batch_loss = total_batch_loss / valid_samples
                total_loss += avg_batch_loss
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_batch_loss:.4f} "
                          f"(valid: {valid_samples}/{batch_size})")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train PharmolixFM")
    parser.add_argument("--config", type=str, default="configs/pharmolix_fm.yaml")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_dummy", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    
    args = parser.parse_args()
    
    print(f"Training PharmolixFM with batch_size={args.batch_size}...")
    print(f"Output: {args.output_dir}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    model = PharmolixFM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.use_dummy:
        print("Using dummy dataset for testing")
        train_dataset = DummyDataset(num_samples=1000)
    else:
        train_dataset = PocketMoleculeDataset(
            data_dir=args.data_dir,
            split="train",
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config.__dict__,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
