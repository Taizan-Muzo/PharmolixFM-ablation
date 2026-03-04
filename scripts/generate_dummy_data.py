#!/usr/bin/env python3
"""
生成虚拟数据集用于训练
模拟官方数据集的规模和结构
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem
from data.molecule import Molecule, Protein, Pocket


def generate_dummy_protein(num_residues=50):
    """生成虚拟蛋白质"""
    # 创建一个简单的肽链
    sequence = ['ALA'] * num_residues
    
    # 创建 PDB 格式的字符串
    pdb_lines = ["HEADER    DUMMY PROTEIN"]
    atom_idx = 1
    res_idx = 1
    
    coords = []
    
    for res in sequence:
        # 简化的原子坐标
        x, y, z = np.random.randn(3) * 10 + [atom_idx * 3.8, 0, 0]
        coords.append([x, y, z])
        
        # N 原子
        pdb_lines.append(f"ATOM{atom_idx:6d}  N   {res} A{res_idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           N")
        atom_idx += 1
        
        # CA 原子
        x, y, z = x + 1.5, y + 0.5, z
        pdb_lines.append(f"ATOM{atom_idx:6d}  CA  {res} A{res_idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
        atom_idx += 1
        coords.append([x, y, z])
        
        # C 原子
        x, y, z = x + 1.5, y - 0.5, z
        pdb_lines.append(f"ATOM{atom_idx:6d}  C   {res} A{res_idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
        atom_idx += 1
        coords.append([x, y, z])
        
        # O 原子
        x, y, z = x, y - 1.0, z + 0.5
        pdb_lines.append(f"ATOM{atom_idx:6d}  O   {res} A{res_idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           O")
        atom_idx += 1
        coords.append([x, y, z])
        
        res_idx += 1
    
    pdb_lines.append("END")
    return "\n".join(pdb_lines)


def generate_dummy_ligand(num_atoms=15):
    """生成虚拟配体分子"""
    # 创建随机分子
    mol = Chem.RWMol()
    
    # 添加原子 (C, N, O)
    atom_types = [6, 7, 8]  # C, N, O
    atom_weights = [0.7, 0.2, 0.1]
    
    for _ in range(num_atoms):
        atom_num = int(np.random.choice(atom_types, p=atom_weights))
        atom = Chem.Atom(atom_num)
        mol.AddAtom(atom)
    
    # 添加随机键
    num_bonds = int(num_atoms * 1.2)
    for _ in range(num_bonds):
        i = int(np.random.randint(0, num_atoms))
        j = int(np.random.randint(0, num_atoms))
        if i != j and mol.GetBondBetweenAtoms(i, j) is None:
            bond_type = np.random.choice([
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.AROMATIC
            ], p=[0.7, 0.2, 0.1])
            try:
                mol.AddBond(i, j, bond_type)
            except:
                pass
    
    # 生成3D坐标
    mol = mol.GetMol()
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass
    
    return mol


def create_dummy_crossdocked_dataset(num_train=1000, num_val=100, num_test=100, output_dir="data/dummy_crossdocked"):
    """
    创建虚拟 CrossDocked 格式的数据集
    
    Args:
        num_train: 训练集样本数
        num_val: 验证集样本数
        num_test: 测试集样本数
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy CrossDocked dataset at {output_path}")
    print(f"  Train: {num_train} samples")
    print(f"  Val: {num_val} samples")
    print(f"  Test: {num_test} samples")
    
    split_index = {
        "train": [],
        "test": []
    }
    
    # 生成训练集
    print("\nGenerating training set...")
    for i in tqdm(range(num_train + num_val)):
        dir_name = f"protein_{i:05d}"
        (output_path / dir_name).mkdir(exist_ok=True)
        
        # 生成蛋白质
        protein_pdb = generate_dummy_protein(num_residues=int(np.random.randint(30, 100)))
        with open(output_path / dir_name / "rec.pdb", "w") as f:
            f.write(protein_pdb)
        
        # 生成口袋 (简化版)
        with open(output_path / dir_name / "pocket.pdb", "w") as f:
            f.write(protein_pdb[:1000])  # 只取部分原子作为口袋
        
        # 生成分子
        ligand_mol = generate_dummy_ligand(num_atoms=int(np.random.randint(10, 30)))
        
        # 保存为 SDF
        writer = Chem.SDWriter(str(output_path / dir_name / "ligand.sdf"))
        writer.write(ligand_mol)
        writer.close()
        
        # 添加到 split
        if i < num_train:
            split_index["train"].append((f"{dir_name}/pocket.pdb", f"{dir_name}/ligand.sdf"))
        else:
            split_index["test"].append((f"{dir_name}/pocket.pdb", f"{dir_name}/ligand.sdf"))
    
    # 生成测试集
    print("\nGenerating test set...")
    for i in range(num_test):
        dir_name = f"test_{i:05d}"
        (output_path / "test_set" / dir_name).mkdir(parents=True, exist_ok=True)
        
        # 生成蛋白质
        protein_pdb = generate_dummy_protein(num_residues=int(np.random.randint(30, 100)))
        with open(output_path / "test_set" / dir_name / f"{dir_name}_rec.pdb", "w") as f:
            f.write(protein_pdb)
        
        # 生成分子
        ligand_mol = generate_dummy_ligand(num_atoms=int(np.random.randint(10, 30)))
        
        # 保存为 SDF
        writer = Chem.SDWriter(str(output_path / "test_set" / dir_name / f"{dir_name}_ligand.sdf"))
        writer.write(ligand_mol)
        writer.close()
    
    # 保存 split 索引
    torch_save = {
        "train": split_index["train"],
        "test": split_index["test"]
    }
    import torch
    torch.save(torch_save, output_path / "split_by_name.pt")
    
    print(f"\n✓ Dummy CrossDocked dataset created at {output_path}")
    print(f"  Total samples: {num_train + num_val + num_test}")
    
    return output_path


def create_dummy_pdbbind_dataset(num_samples=500, output_dir="data/dummy_pdbbind"):
    """
    创建虚拟 PDBbind 格式的数据集
    
    Args:
        num_samples: 样本数
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy PDBbind dataset at {output_path}")
    print(f"  Samples: {num_samples}")
    
    # 创建目录结构
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "val").mkdir(exist_ok=True)
    (output_path / "test").mkdir(exist_ok=True)
    
    splits = {
        "train": int(num_samples * 0.8),
        "val": int(num_samples * 0.1),
        "test": int(num_samples * 0.1)
    }
    
    idx = 0
    for split, num in splits.items():
        print(f"\nGenerating {split} set ({num} samples)...")
        for _ in tqdm(range(num)):
            dir_name = f"{split}_{idx:04d}"
            (output_path / split / dir_name).mkdir(exist_ok=True)
            
            # 生成蛋白质
            protein_pdb = generate_dummy_protein(num_residues=int(np.random.randint(30, 100)))
            with open(output_path / split / dir_name / "protein.pdb", "w") as f:
                f.write(protein_pdb)
            
            # 生成分子
            ligand_mol = generate_dummy_ligand(num_atoms=int(np.random.randint(10, 30)))
            
            # 保存为 SDF
            writer = Chem.SDWriter(str(output_path / split / dir_name / "ligand.sdf"))
            writer.write(ligand_mol)
            writer.close()
            
            idx += 1
    
    print(f"\n✓ Dummy PDBbind dataset created at {output_path}")
    print(f"  Total samples: {sum(splits.values())}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate dummy datasets")
    parser.add_argument("--dataset", type=str, 
                       choices=["crossdocked", "pdbbind", "all"],
                       default="all",
                       help="Dataset to generate")
    parser.add_argument("--num_train", type=int, default=1000,
                       help="Number of training samples (for CrossDocked)")
    parser.add_argument("--num_val", type=int, default=100,
                       help="Number of validation samples")
    parser.add_argument("--num_test", type=int, default=100,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="data/",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Generating Dummy Datasets")
    print("=" * 70)
    
    if args.dataset in ["crossdocked", "all"]:
        create_dummy_crossdocked_dataset(
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            output_dir=Path(args.output_dir) / "dummy_crossdocked"
        )
    
    if args.dataset in ["pdbbind", "all"]:
        total = args.num_train + args.num_val + args.num_test
        create_dummy_pdbbind_dataset(
            num_samples=total,
            output_dir=Path(args.output_dir) / "dummy_pdbbind"
        )
    
    print("\n" + "=" * 70)
    print("Dummy dataset generation completed!")
    print("=" * 70)
    
    # 显示状态
    print("\nDataset status:")
    import subprocess
    result = subprocess.run(
        ["python", "scripts/download_all_data.py", "--dataset", "status"],
        capture_output=True,
        text=True
    )
    print(result.stdout)


if __name__ == "__main__":
    from tqdm import tqdm
    main()
