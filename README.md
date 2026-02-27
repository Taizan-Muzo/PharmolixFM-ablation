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

### 训练
```bash
python scripts/train.py --config configs/pharmolix_fm.yaml
```

### 推理（对接）
```bash
python scripts/inference.py --checkpoint model.ckpt --pdb protein.pdb --sdf ligand.sdf --task docking
```

### 推理（分子生成）
```bash
python scripts/inference.py --checkpoint model.ckpt --pdb protein.pdb --sdf ref_ligand.sdf --task generation
```

## 目录结构
```
PharmolixFM-ablation/
├── models/           # 模型定义（精简版）
├── data/             # 数据加载器
├── configs/          # 配置文件
├── scripts/          # 训练和评估脚本
└── utils/            # 工具函数
```

## 许可证
MIT

---
**免责声明**: 本仓库为研究用途的精简版本，非 PharMolix 官方维护。如需完整功能，请使用 [OpenBioMed](https://github.com/PharMolix/OpenBioMed)。
