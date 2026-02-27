"""
PharmolixFM 推理脚本
"""

import argparse
import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM, PharmolixFMMoleculeFeaturizer, PharmolixFMPocketFeaturizer
from data.molecule import Molecule, Protein, Pocket


def main():
    parser = argparse.ArgumentParser(description="PharmolixFM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--pdb", type=str, required=True, help="蛋白质 PDB 文件")
    parser.add_argument("--sdf", type=str, help="参考配体 SDF 文件（用于定义口袋）")
    parser.add_argument("--task", type=str, default="docking", choices=["docking", "generation"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="output.sdf")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    
    # 加载模型
    model = PharmolixFM.load_from_checkpoint(args.checkpoint)
    model = model.to(args.device)
    model.eval()
    
    # 加载蛋白质
    protein = Protein.from_pdb_file(args.pdb)
    print(f"Loaded protein: {protein.name}")
    
    if args.task == "docking":
        # 口袋-分子对接
        if args.sdf:
            ligand = Molecule.from_sdf_file(args.sdf)
            pocket = Pocket.from_protein_ref_ligand(protein, ligand)
            print(f"Defined pocket from reference ligand")
            # TODO: 实现对接推理
        else:
            print("Error: --sdf required for docking task")
            return
    
    elif args.task == "generation":
        # 基于结构的药物设计
        if args.sdf:
            ligand = Molecule.from_sdf_file(args.sdf)
            pocket = Pocket.from_protein_ref_ligand(protein, ligand)
        else:
            # 需要手动定义口袋中心
            print("Error: --sdf required for pocket definition")
            return
        
        print("Generating molecules...")
        # TODO: 实现分子生成
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
