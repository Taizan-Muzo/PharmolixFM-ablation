"""
PharmolixFM 评估脚本
"""

import argparse
import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pharmolix_fm import PharmolixFM


def main():
    parser = argparse.ArgumentParser(description="Evaluate PharmolixFM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="eval_results.json")
    
    args = parser.parse_args()
    
    print(f"Evaluating model: {args.checkpoint}")
    
    # 加载模型
    model = PharmolixFM.load_from_checkpoint(args.checkpoint)
    model = model.to(args.device)
    model.eval()
    
    # TODO: 实现评估逻辑
    # 1. 加载测试数据
    # 2. 运行推理
    # 3. 计算指标 (RMSD, 亲和力等)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
