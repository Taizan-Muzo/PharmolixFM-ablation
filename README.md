# PharmolixFM-ablation

**⚠️ 本仓库仅用于简约消融实验，非官方实现**

PharmolixFM 的精简版本，提取自 [OpenBioMed](https://github.com/PharMolix/OpenBioMed)，用于独立的消融实验研究。

## 原论文

**PharmolixFM: All-Atom Molecular Foundation Model for Pocket-Molecule Docking and Structure-Based Drug Design**

- 📃 Paper: [arXiv:2503.21788](https://arxiv.org/abs/2503.21788)
- 🏢 Authors: PharMolix Inc. & Institute of AI Industry Research (AIR), Tsinghua University
- 🔗 Original Code: [OpenBioMed](https://github.com/PharMolix/OpenBioMed)

### Citation
```bibtex
@article{zhu2025pharmolixfm,
  title={PharmolixFM: All-Atom Molecular Foundation Model for Pocket-Molecule Docking and Structure-Based Drug Design},
  author={Zhu, Yinjie and others},
  journal={arXiv preprint arXiv:2503.21788},
  year={2025}
}
```

## 功能
- ✅ Pocket-Molecule Docking（口袋-分子对接）
- ✅ Structure-Based Drug Design（基于结构的药物设计）

## 与原版的区别
| 特性 | 原版 (OpenBioMed) | 本仓库 (ablation) |
|------|------------------|------------------|
| 依赖 | 完整依赖链 | 精简依赖 |
| 功能 | 全功能 | 仅核心功能 |
| 用途 | 生产环境 | 消融实验 |
| 维护 | 官方维护 | 实验用 |

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 下载测试数据
```bash
python scripts/download_data.py --dataset test
```

### 2. 训练（使用虚拟数据测试）
```bash
python scripts/train.py --use_dummy --epochs 5
```

### 3. 训练（使用真实数据）
```bash
# 先下载 PDBbind 或 CrossDocked 数据集
python scripts/download_data.py --dataset pdbbind

# 然后训练
python scripts/train.py --data_dir data/pdbbind/
```

### 4. 推理
```bash
python scripts/inference.py \
    --checkpoint checkpoints/final_model.pt \
    --pdb data/test_examples/4XLI.pdb \
    --sdf data/test_examples/ligand.sdf \
    --task docking
```

### 5. 评估
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/final_model.pt \
    --test_data data/test/ \
    --output eval_results.json
```

## 目录结构
```
PharmolixFM-ablation/
├── models/           # 模型定义（精简版）
│   └── pharmolix_fm.py
├── data/             # 数据加载器
│   ├── molecule.py
│   └── dataset.py
├── utils/            # 工具函数
│   ├── config.py
│   ├── featurizer.py
│   └── pocket_featurizer.py
├── scripts/          # 训练和评估脚本
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── download_data.py
└── configs/          # 配置文件
    └── pharmolix_fm.yaml
```

## 已知限制

1. **训练循环不完整**：BFN 损失函数未完全实现
2. **数据加载简化**：口袋原子解析需要进一步完善
3. **评估指标缺失**：RMSD、亲和力等指标待实现
4. **无预训练权重**：需要从头训练或使用 OpenBioMed 的权重

## 开发计划

- [ ] 实现完整的 BFN 训练损失
- [ ] 添加口袋原子解析（从 PDB）
- [ ] 实现评估指标（RMSD、亲和力）
- [ ] 支持加载 OpenBioMed 预训练权重
- [ ] 添加更多数据集支持

## 许可证
MIT

---
**免责声明**: 本仓库为研究用途的精简版本，非 PharMolix 官方维护。如需完整功能，请使用 [OpenBioMed](https://github.com/PharMolix/OpenBioMed)。
