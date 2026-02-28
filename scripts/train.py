"""
PharmolixFM 训练脚本（真正的 batch 并行）
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM
from utils.config import Config
from data.dataset import DummyDataset, PocketMoleculeDataset


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


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个 epoch - 使用 batch 并行"""
    model.train()
    total_loss = 0.0
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
            # 使用 batch 版本前向传播（真正的并行）
            loss_dict = model.forward_pocket_molecule_docking_batch(pockets, molecules)
            loss = loss_dict['loss']
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                batch_size = molecules['batch'].max().item() + 1
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f} (batch_size={batch_size})")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train PharmolixFM")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_dummy", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    
    args = parser.parse_args()
    
    print(f"Training PharmolixFM with batch_size={args.batch_size}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    model = PharmolixFM(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    dataset = DummyDataset(num_samples=1000) if args.use_dummy else \
              PocketMoleculeDataset(data_dir=args.data_dir, split="train")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                           collate_fn=collate_fn, num_workers=0)
    
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")
    
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    print(f"Training completed!")


if __name__ == "__main__":
    main()
