"""
PharmolixFM 分布式训练脚本 (DDP)
使用 PyTorch DistributedDataParallel 实现多 GPU 训练
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM
from utils.config import Config
from utils.config_official import OfficialConfig
from data.dataset import DummyDataset, PocketMoleculeDataset


def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def collate_fn(batch):
    """批次整理"""
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


def train_epoch_ddp(model, dataloader, optimizer, rank, epoch):
    """DDP 训练一个 epoch"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (molecules, pockets, metadata) in enumerate(dataloader):
        # 移到设备
        for key in molecules:
            if isinstance(molecules[key], torch.Tensor):
                molecules[key] = molecules[key].to(rank)
        for key in pockets:
            if isinstance(pockets[key], torch.Tensor):
                pockets[key] = pockets[key].to(rank)
        
        optimizer.zero_grad()
        
        try:
            loss_dict = model.module.forward_pocket_molecule_docking_batch(pockets, molecules)
            loss = loss_dict['loss']
            
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0 and rank == 0:
                batch_size = molecules['batch'].max().item() + 1
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f} (batch_size={batch_size})")
        
        except Exception as e:
            if rank == 0:
                print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # 同步所有进程的损失
    avg_loss = torch.tensor([total_loss / max(num_batches, 1)], device=rank)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    
    return avg_loss.item()


def validate_ddp(model, dataloader, rank):
    """DDP 验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (molecules, pockets, metadata) in enumerate(dataloader):
            # 移到设备
            for key in molecules:
                if isinstance(molecules[key], torch.Tensor):
                    molecules[key] = molecules[key].to(rank)
            for key in pockets:
                if isinstance(pockets[key], torch.Tensor):
                    pockets[key] = pockets[key].to(rank)
            
            try:
                loss_dict = model.module.forward_pocket_molecule_docking_batch(pockets, molecules)
                loss = loss_dict['loss']
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
            except Exception as e:
                continue
    
    # 同步所有进程的损失
    avg_loss = torch.tensor([total_loss / max(num_batches, 1)], device=rank)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    
    return avg_loss.item()


def train_worker(rank, world_size, args):
    """每个 GPU 上的训练 worker"""
    setup_distributed(rank, world_size)
    
    # 创建模型
    if args.use_official_config:
        config = OfficialConfig()
        if rank == 0:
            print("Using official model configuration")
    else:
        config = Config()
        if rank == 0:
            print("Using default model configuration")
    
    model = PharmolixFM(config).to(rank)
    
    # 加载预训练权重
    if args.pretrain_checkpoint and Path(args.pretrain_checkpoint).exists():
        if rank == 0:
            print(f"Loading pretrain checkpoint: {args.pretrain_checkpoint}")
        
        ckpt = torch.load(args.pretrain_checkpoint, map_location=f'cuda:{rank}')
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        
        if rank == 0:
            print("Pretrain checkpoint loaded")
    
    # 包装为 DDP 模型
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=(rank == 0)
    )
    
    # 创建数据集
    if args.use_dummy:
        if rank == 0:
            print("Using dummy dataset")
        train_dataset = DummyDataset(num_samples=1000)
        val_dataset = DummyDataset(num_samples=100)
    else:
        if rank == 0:
            print(f"Loading data from: {args.data_dir}")
        train_dataset = PocketMoleculeDataset(data_dir=args.data_dir, split="train")
        val_dataset = PocketMoleculeDataset(data_dir=args.data_dir, split="val")
    
    # 创建 DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"\nTraining with {world_size} GPUs")
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 训练循环
    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print("-" * 40)
        
        # 设置 epoch 以确保不同进程使用不同的数据
        train_sampler.set_epoch(epoch)
        
        # 训练
        train_loss = train_epoch_ddp(model, train_loader, optimizer, rank, epoch)
        
        # 验证
        val_loss = validate_ddp(model, val_loader, rank)
        
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存检查点 (只在 rank 0 上)
        if rank == 0 and epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # 保存最佳模型
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.module.state_dict(), output_dir / "best_model.pt")
            print(f"Best model saved (val_loss: {val_loss:.4f})")
    
    # 保存最终模型
    if rank == 0:
        torch.save(model.module.state_dict(), output_dir / "final_model.pt")
        print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train PharmolixFM with DDP")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ddp_run/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--use_dummy", action="store_true")
    parser.add_argument("--use_official_config", action="store_true",
                       help="Use official model configuration (requires more GPU memory)")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None,
                       help="Path to pretrain checkpoint (e.g., checkpoints/openbiomed/pharmolix_fm_converted.pt)")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--world_size", type=int, default=None,
                       help="Number of GPUs (default: all available)")
    
    args = parser.parse_args()
    
    # 确定 world_size
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
    
    if args.world_size < 1:
        print("No GPU available! Falling back to CPU training.")
        args.world_size = 1
    
    print(f"Starting DDP training with {args.world_size} GPUs...")
    
    # 启动多进程训练
    mp.spawn(
        train_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
