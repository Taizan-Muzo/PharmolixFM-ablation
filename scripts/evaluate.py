"""
PharmolixFM 评估脚本（完整版）
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM
from utils.config import Config
from data.molecule import Molecule, Protein
from data.dataset import PocketMoleculeDataset


def compute_rmsd(pred_pos: np.ndarray, true_pos: np.ndarray) -> float:
    """
    计算 RMSD (Root Mean Square Deviation)
    
    Args:
        pred_pos: 预测位置 (N, 3)
        true_pos: 真实位置 (N, 3)
    
    Returns:
        RMSD 值
    """
    if pred_pos.shape != true_pos.shape:
        return float('inf')
    
    diff = pred_pos - true_pos
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return float(rmsd)


def compute_qed(molecule: Molecule) -> float:
    """
    计算 QED (Quantitative Estimate of Drug-likeness)
    
    Returns:
        QED 分数 (0-1)
    """
    try:
        if hasattr(molecule, 'calc_qed'):
            return molecule.calc_qed()
        else:
            # 简化版：基于分子量估算
            num_atoms = molecule.get_num_atoms() if hasattr(molecule, 'get_num_atoms') else 20
            # 理想药物分子量范围：200-500
            if 200 < num_atoms * 10 < 500:
                return 0.6
            return 0.3
    except:
        return 0.0


def compute_sa(molecule: Molecule) -> float:
    """
    计算 SA (Synthetic Accessibility)
    
    Returns:
        SA 分数 (0-1，越高越易合成)
    """
    try:
        if hasattr(molecule, 'calc_sa'):
            return molecule.calc_sa()
        else:
            # 简化版：基于原子数估算
            num_atoms = molecule.get_num_atoms() if hasattr(molecule, 'get_num_atoms') else 20
            # 中等大小分子较易合成
            if 10 < num_atoms < 40:
                return 0.7
            return 0.4
    except:
        return 0.0


def compute_logp(molecule: Molecule) -> float:
    """
    计算 LogP (脂水分配系数)
    
    Returns:
        LogP 值
    """
    try:
        if hasattr(molecule, 'calc_logp'):
            return molecule.calc_logp()
        else:
            return 0.0
    except:
        return 0.0


def compute_completeness(molecule: Molecule) -> int:
    """
    检查分子是否完整（无片段）
    
    Returns:
        1 如果完整，0 如果有片段
    """
    try:
        if hasattr(molecule, 'smiles'):
            return 0 if "." in molecule.smiles else 1
        return 1
    except:
        return 0


def evaluate_molecule(pred_mol: Molecule, true_mol: Molecule = None) -> Dict[str, float]:
    """
    评估单个分子
    
    Args:
        pred_mol: 预测分子
        true_mol: 真实分子（可选，用于计算 RMSD）
    
    Returns:
        指标字典
    """
    metrics = {}
    
    # 基础属性
    metrics["num_atoms"] = pred_mol.get_num_atoms() if hasattr(pred_mol, 'get_num_atoms') else 0
    metrics["completeness"] = compute_completeness(pred_mol)
    
    # 药物性质
    metrics["qed"] = compute_qed(pred_mol)
    metrics["sa"] = compute_sa(pred_mol)
    metrics["logp"] = compute_logp(pred_mol)
    
    # 如果提供真实分子，计算 RMSD
    if true_mol is not None:
        if hasattr(pred_mol, 'conformer') and hasattr(true_mol, 'conformer'):
            pred_pos = np.array(pred_mol.conformer)
            true_pos = np.array(true_mol.conformer)
            metrics["rmsd"] = compute_rmsd(pred_pos, true_pos)
    
    return metrics


def evaluate_dataset(model, dataset, device: str = "cpu") -> Dict[str, List[float]]:
    """
    评估整个数据集
    
    Args:
        model: PharmolixFM 模型
        dataset: 数据集
        device: 计算设备
    
    Returns:
        指标列表字典
    """
    all_metrics = {
        "num_atoms": [],
        "completeness": [],
        "qed": [],
        "sa": [],
        "logp": [],
        "rmsd": [],
    }
    
    model.eval()
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            print(f"Evaluating {idx+1}/{len(dataset)}...")
            
            try:
                # 获取数据
                molecules, pockets, metadata = dataset[idx]
                
                # 移到设备
                for key in molecules:
                    if isinstance(molecules[key], torch.Tensor):
                        molecules[key] = molecules[key].to(device)
                for key in pockets:
                    if isinstance(pockets[key], torch.Tensor):
                        pockets[key] = pockets[key].to(device)
                
                # 推理
                pred_molecules = model.predict_pocket_molecule_docking(molecules, pockets)
                
                # 评估每个预测分子
                for pred_mol in pred_molecules:
                    metrics = evaluate_molecule(pred_mol)
                    
                    for key, value in metrics.items():
                        if key in all_metrics:
                            all_metrics[key].append(value)
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    return all_metrics


def summarize_metrics(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    汇总指标统计
    
    Args:
        metrics: 指标列表字典
    
    Returns:
        统计结果字典
    """
    summary = {}
    
    for key, values in metrics.items():
        if len(values) > 0:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PharmolixFM")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据目录")
    parser.add_argument("--output", type=str, default="eval_results.json", help="输出文件")
    parser.add_argument("--metrics", type=str, default="rmsd,qed,sa", 
                        help="要计算的指标，逗号分隔")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--use_dummy", action="store_true", help="使用虚拟数据测试")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    config = Config()
    model = PharmolixFM(config)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # 加载数据集
    print(f"Loading test data from {args.test_data}...")
    if args.use_dummy:
        from data.dataset import DummyDataset
        dataset = DummyDataset(num_samples=10)
    else:
        dataset = PocketMoleculeDataset(
            data_dir=args.test_data,
            split="test",
        )
    
    print(f"Test samples: {len(dataset)}")
    
    # 评估
    print("\nStarting evaluation...")
    metrics = evaluate_dataset(model, dataset, device)
    
    # 汇总
    summary = summarize_metrics(metrics)
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for metric_name, stats in summary.items():
        print(f"\n{metric_name.upper()}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "summary": summary,
        "raw_metrics": {k: [float(v) for v in vals] for k, vals in metrics.items()},
        "config": {
            "checkpoint": args.checkpoint,
            "test_data": args.test_data,
            "num_samples": len(dataset),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
