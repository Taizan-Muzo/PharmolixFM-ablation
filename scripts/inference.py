"""
PharmolixFM 推理脚本（完整版）
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM
from utils.config import Config
from data.molecule import Molecule, Protein, Pocket


def load_model(checkpoint_path: str, device: str = "cpu"):
    """加载模型"""
    config = Config()
    model = PharmolixFM(config)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def docking_inference(model, pdb_path: str, sdf_path: str, device: str = "cpu"):
    """
    口袋-分子对接推理
    
    Args:
        model: PharmolixFM 模型
        pdb_path: 蛋白质 PDB 文件路径
        sdf_path: 配体 SDF 文件路径
        device: 计算设备
    
    Returns:
        对接后的分子列表
    """
    print(f"Loading protein from {pdb_path}...")
    protein = Protein.from_pdb_file(pdb_path)
    
    print(f"Loading ligand from {sdf_path}...")
    ligand = Molecule.from_sdf_file(sdf_path)
    
    print("Defining pocket...")
    pocket = Pocket.from_protein_ref_ligand(protein, ligand)
    
    # 特征化
    print("Featurizing...")
    featurizers = model.featurizers
    mol_features = featurizers["molecule"](ligand)
    pocket_features = featurizers["pocket"](pocket)
    
    # 添加 batch 属性（单样本推理，所有节点属于批次 0）
    import torch
    mol_features.batch = torch.zeros(mol_features.pos.shape[0], dtype=torch.long, device=device)
    pocket_features.batch = torch.zeros(pocket_features.pos.shape[0], dtype=torch.long, device=device)
    
    # 添加 {key}_batch 属性用于后处理
    mol_features.pos_batch = mol_features.batch
    mol_features.node_type_batch = mol_features.batch
    mol_features.halfedge_type_batch = torch.zeros(mol_features.halfedge_type.shape[0], dtype=torch.long, device=device)
    
    # 移到设备
    for key in mol_features.keys():
        if isinstance(mol_features[key], torch.Tensor) and mol_features[key].device != device:
            mol_features[key] = mol_features[key].to(device)
    for key in pocket_features.keys():
        if isinstance(pocket_features[key], torch.Tensor) and pocket_features[key].device != device:
            pocket_features[key] = pocket_features[key].to(device)
    
    # 推理
    print("Running docking inference...")
    with torch.no_grad():
        results = model.predict_pocket_molecule_docking(mol_features, pocket_features)
    
    return results


def design_inference(model, pdb_path: str, num_samples: int = 10, device: str = "cpu"):
    """
    基于结构的药物设计推理
    
    Args:
        model: PharmolixFM 模型
        pdb_path: 蛋白质 PDB 文件路径（需包含参考配体定义口袋）
        num_samples: 生成样本数
        device: 计算设备
    
    Returns:
        生成的分子列表
    """
    print(f"Loading protein from {pdb_path}...")
    protein = Protein.from_pdb_file(pdb_path)
    
    # 创建口袋（使用整个蛋白质作为口袋，以蛋白质中心为口袋中心）
    print("Creating pocket...")
    from Bio.PDB import PDBParser
    import numpy as np
    
    pocket = Pocket()
    pocket.protein = protein
    
    # 解析 PDB 文件提取所有原子
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # 氨基酸映射
    aa_to_idx = {
        'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
        'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
        'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
        'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
    }
    
    # 原子序数映射
    element_to_z = {
        'C': 6, 'N': 7, 'O': 8, 'S': 16, 'H': 1,
        'P': 15, 'F': 9, 'Cl': 17, 'CL': 17, 'Br': 35, 'BR': 35, 'I': 53,
    }
    
    # 骨架原子
    backbone_atoms = {'N', 'CA', 'C', 'O'}
    
    pocket.atoms = []
    coordinates = []
    
    for struct_model in structure:
        for chain in struct_model:
            for residue in chain:
                res_name = residue.resname
                aa_type = aa_to_idx.get(res_name, 0)
                
                for atom in residue:
                    coord = atom.coord
                    atom_name = atom.name
                    element = atom.element if atom.element else atom_name[0]
                    atomic_number = element_to_z.get(element, 6)
                    is_backbone = 1 if atom_name in backbone_atoms else 0
                    
                    pocket.atoms.append({
                        'atomic_number': atomic_number,
                        'aa_type': aa_type,
                        'is_backbone': is_backbone,
                        'name': atom_name,
                        'residue': res_name,
                    })
                    coordinates.append(coord)
    
    if coordinates:
        pocket.conformer = np.array(coordinates)
        pocket.center = np.mean(coordinates, axis=0)
    else:
        pocket.center = np.array([0.0, 0.0, 0.0])
    
    # 特征化
    print("Featurizing...")
    featurizers = model.featurizers
    pocket_features = featurizers["pocket"](pocket)
    
    # 添加 batch 属性
    import torch
    pocket_features.batch = torch.zeros(pocket_features.pos.shape[0], dtype=torch.long, device=device)
    
    # 移到设备
    for key in pocket_features.keys():
        if isinstance(pocket_features[key], torch.Tensor) and pocket_features[key].device != device:
            pocket_features[key] = pocket_features[key].to(device)
    
    # 生成多个分子
    print(f"Generating {num_samples} molecules...")
    all_molecules = []
    
    with torch.no_grad():
        for i in range(num_samples):
            print(f"  Sample {i+1}/{num_samples}...")
            molecules = model.predict_structure_based_drug_design(pocket_features)
            all_molecules.extend(molecules)
    
    return all_molecules


def save_molecules(molecules, output_path: str):
    """保存分子到 SDF 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 简化版：保存为文本格式
    # 实际应该使用 RDKit 保存为 SDF
    with open(output_path, 'w') as f:
        f.write(f"# Generated {len(molecules)} molecules\n")
        for i, mol in enumerate(molecules):
            f.write(f"\n>>> Molecule {i+1} <<<\n")
            if hasattr(mol, 'smiles'):
                f.write(f"SMILES: {mol.smiles}\n")
            if hasattr(mol, 'get_num_atoms'):
                f.write(f"Num atoms: {mol.get_num_atoms()}\n")
    
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PharmolixFM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--pdb", type=str, required=True, help="蛋白质 PDB 文件")
    parser.add_argument("--sdf", type=str, help="配体 SDF 文件（对接任务）")
    parser.add_argument("--task", type=str, choices=["docking", "design"], 
                        default="docking", help="推理任务")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="生成样本数（设计任务）")
    parser.add_argument("--output", type=str, default="output.sdf", 
                        help="输出文件路径")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="计算设备 (cpu/cuda:0)")
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    if not os.path.exists(args.pdb):
        print(f"Error: PDB file not found: {args.pdb}")
        return
    if args.task == "docking" and not args.sdf:
        print("Error: --sdf required for docking task")
        return
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    print(f"Model loaded on {device}")
    
    # 推理
    if args.task == "docking":
        results = docking_inference(model, args.pdb, args.sdf, device)
    else:  # design
        results = design_inference(model, args.pdb, args.num_samples, device)
    
    # 保存结果
    print(f"\nGenerated {len(results)} molecules")
    save_molecules(results, args.output)
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
