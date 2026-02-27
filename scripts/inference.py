"""
PharmolixFM 推理脚本（修复版）
"""

import argparse
import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM, PharmolixFMMoleculeFeaturizer, PharmolixFMPocketFeaturizer
from utils.config import Config
from data.molecule import Molecule, Protein, Pocket


def main():
    parser = argparse.ArgumentParser(description="PharmolixFM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径 (.pt 或 .pth)")
    parser.add_argument("--pdb", type=str, required=True, help="蛋白质 PDB 文件")
    parser.add_argument("--sdf", type=str, help="参考配体 SDF 文件（用于定义口袋）")
    parser.add_argument("--task", type=str, default="docking", choices=["docking", "generation"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="output.sdf")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    
    # 创建配置
    config = Config()
    
    # 创建模型
    model = PharmolixFM(config)
    
    # 加载权重（修复：使用 torch.load 而非 load_from_checkpoint）
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    print(f"Model loaded successfully!")
    
    # 加载蛋白质
    protein = Protein.from_pdb_file(args.pdb)
    print(f"Loaded protein: {protein.name}")
    
    if args.task == "docking":
        if not args.sdf:
            print("Error: --sdf required for docking task")
            return
        
        ligand = Molecule.from_sdf_file(args.sdf)
        pocket = Pocket.from_protein_ref_ligand(protein, ligand)
        print(f"Defined pocket from reference ligand")
        
        # TODO: 实现特征化和推理
        print("Docking inference not yet implemented")
    
    elif args.task == "generation":
        if args.sdf:
            ligand = Molecule.from_sdf_file(args.sdf)
            pocket = Pocket.from_protein_ref_ligand(protein, ligand)
        else:
            print("Error: --sdf required for pocket definition")
            return
        
        print("Molecule generation not yet implemented")
    
    print(f"Results would be saved to {args.output}")


if __name__ == "__main__":
    main()
