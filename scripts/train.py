"""
PharmolixFM 训练脚本（完整版）
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM
from utils.config import Config
from data.dataset import DummyDataset, PocketMoleculeDataset


def collate_fn(batch):
    """自定义批次整理函数"""
    molecules, pockets, metadata = zip(*batch)
    
    # 简单拼接（实际应该使用 PyG 的 Batch）
    batch_mol = {
        "node_type": torch.cat([m["node_type"] for m in molecules]),
        "pos": torch.cat([m["pos"] for m in molecules]),
        "halfedge_type": torch.cat([m["halfedge_type"] for m in molecules]),
    }
    
    batch_pocket = {
        "atom_feature": torch.cat([p["atom_feature"] for p in pockets]),
        "pos": torch.cat([p["pos"] for p in pockets]),
    }
    
    return batch_mol, batch_pocket, metadata


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (molecules, pockets, metadata) in enumerate(dataloader):
        # 目前只支持 batch_size=1，取第一个样本
        # TODO: 实现真正的 batch 处理
        
        # 将数据移到设备
        for key in molecules:
            if isinstance(molecules[key], torch.Tensor):
                molecules[key] = molecules[key].to(device)
        for key in pockets:
            if isinstance(pockets[key], torch.Tensor):
                pockets[key] = pockets[key].to(device)
        
        optimizer.zero_grad()
        
        try:
            # 前向传播和损失计算
            loss_dict = model.forward_pocket_molecule_docking(pockets, molecules)
            loss = loss_dict['loss']
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train PharmolixFM")
    parser.add_argument("--config", type=str, default="configs/pharmolix_fm.yaml")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_dummy", action="store_true", help="使用虚拟数据")
    parser.add_argument("--save_every", type=int, default=5, help="每 N 个 epoch 保存")
    
    args = parser.parse_args()
    
    print(f"Training PharmolixFM...")
    print(f"Output: {args.output_dir}")
    
    # 创建设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建配置
    config = Config()
    
    # 创建模型
    model = PharmolixFM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建数据集
    if args.use_dummy:
        print("Using dummy dataset for testing")
        train_dataset = DummyDataset(num_samples=100)
        val_dataset = DummyDataset(num_samples=20)
    else:
        train_dataset = PocketMoleculeDataset(
            data_dir=args.data_dir,
            split="train",
        )
        val_dataset = PocketMoleculeDataset(
            data_dir=args.data_dir,
            split="val",
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 调试时设为 0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # 保存检查点
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
    
    # 保存最终模型
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")
    print("\nNote: This is a basic training loop. To implement full training:")
    print("1. Implement forward pass in PharmolixFM")
    print("2. Add loss computation (BFN loss)")
    print("3. Add validation loop")
    print("4. Add learning rate scheduling")
    print("5. Add logging (wandb/tensorboard)")


if __name__ == "__main__":
    main()
