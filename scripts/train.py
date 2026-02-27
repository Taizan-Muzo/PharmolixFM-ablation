"""
PharmolixFM 训练脚本（完整版，支持 Batch）
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
    """自定义批次整理函数"""
    molecules, pockets, metadata = zip(*batch)
    
    batch_mol = {
        "node_type": torch.cat([m["node_type"] for m in molecules]),
        "pos": torch.cat([m["pos"] for m in molecules]),
        "halfedge_type": torch.cat([m["halfedge_type"] for m in molecules]),
        "batch": torch.cat([torch.full((len(m["pos"]),), i, dtype=torch.long) for i, m in enumerate(molecules)]),
    }
    
    batch_pocket = {
        "atom_feature": torch.cat([p["atom_feature"] for p in pockets]),
        "pos": torch.cat([p["pos"] for p in pockets]),
        "batch": torch.cat([torch.full((len(p["pos"]),), i, dtype=torch.long) for i, p in enumerate(pockets)]),
    }
    
    return batch_mol, batch_pocket, metadata


def split_batch(molecule_batch, pocket_batch, batch_idx):
    """从 batch 中提取单个样本"""
    mol_mask = molecule_batch["batch"] == batch_idx
    molecule = {
        "node_type": molecule_batch["node_type"][mol_mask],
        "pos": molecule_batch["pos"][mol_mask],
        "halfedge_type": molecule_batch["halfedge_type"][mol_mask],
        "halfedge_index": torch.zeros(2, 0, dtype=torch.long, device=molecule_batch["pos"].device),
    }
    
    pocket_mask = pocket_batch["batch"] == batch_idx
    pocket = {
        "atom_feature": pocket_batch["atom_feature"][pocket_mask],
        "pos": pocket_batch["pos"][pocket_mask],
        "knn_edge_index": torch.zeros(2, 0, dtype=torch.long, device=pocket_batch["pos"].device),
    }
    
    return molecule, pocket


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (molecules, pockets, metadata) in enumerate(dataloader):
        batch_size = len(metadata)
        
        for key in molecules:
            if isinstance(molecules[key], torch.Tensor):
                molecules[key] = molecules[key].to(device)
        for key in pockets:
            if isinstance(pockets[key], torch.Tensor):
                pockets[key] = pockets[key].to(device)
        
        batch_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            try:
                molecule, pocket = split_batch(molecules, pockets, i)
                optimizer.zero_grad()
                loss_dict = model.forward_pocket_molecule_docking(pocket, molecule)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                valid_samples += 1
            except Exception as e:
                print(f"  Error in sample {i}: {e}")
                continue
        
        if valid_samples > 0:
            avg_batch_loss = batch_loss / valid_samples
            total_loss += avg_batch_loss
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_batch_loss:.4f}")
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train PharmolixFM")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_dummy", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    model = PharmolixFM(config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.use_dummy:
        train_dataset = DummyDataset(num_samples=100)
    else:
        train_dataset = PocketMoleculeDataset(data_dir=args.data_dir, split="train")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"Train samples: {len(train_dataset)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training completed! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
