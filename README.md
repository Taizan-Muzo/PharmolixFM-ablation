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

## 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 模型架构 | ✅ 完整 | 支持口袋-分子对接 + 药物设计 |
| BFN 损失函数 | ✅ 已实现 | 基于 BFN 论文简化实现 |
| 训练 | ✅ 可用 | 支持虚拟/真实数据训练 |
| 推理 | ✅ 可用 | 支持对接和设计任务推理 |
| 评估 | ⚠️ 基础 | RMSD、亲和力指标待完善 |
| 预训练权重 | ❌ 不支持 | 需从头训练 |

## 与原版的区别

| 特性 | 原版 (OpenBioMed) | 本仓库 (ablation) |
|------|------------------|------------------|
| 依赖 | 完整依赖链 | 精简依赖 |
| 功能 | 全功能 | 核心功能 |
| 训练 | 未开源损失函数 | ✅ BFN 损失已实现 |
| 用途 | 生产环境 | 消融实验 |
| 维护 | 官方维护 | 实验用 |

## 安装

```bash
# 克隆仓库
git clone https://github.com/Taizan-Muzo/PharmolixFM-ablation.git
cd PharmolixFM-ablation

# 安装依赖
pip install -r requirements.txt
```

### 依赖要求
- Python >= 3.8
- PyTorch >= 2.0
- PyTorch Geometric
- RDKit
- NumPy, SciPy, tqdm

## 快速开始

### 1. 测试安装
```bash
python -c "from models.pharmolix_fm import PharmolixFM; print('✓ Import OK')"
```

### 2. 下载测试数据
```bash
python scripts/download_data.py --dataset test
```

### 3. 训练（虚拟数据测试）
```bash
python scripts/train.py --use_dummy --epochs 5 --batch_size 4
```

### 4. 训练（真实数据）
```bash
# 下载 PDBbind 数据集（需手动）
python scripts/download_data.py --dataset pdbbind

# 训练
python scripts/train.py \
    --data_dir data/pdbbind/ \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --output_dir checkpoints/
```

### 5. 推理 - 口袋分子对接
```bash
python scripts/inference.py \
    --checkpoint checkpoints/final_model.pt \
    --pdb data/test_examples/4XLI.pdb \
    --sdf data/test_examples/ligand.sdf \
    --task docking \
    --output output_docked.sdf
```

### 6. 推理 - 基于结构的药物设计
```bash
python scripts/inference.py \
    --checkpoint checkpoints/final_model.pt \
    --pdb data/test_examples/4XLI.pdb \
    --task design \
    --num_samples 10 \
    --output output_molecules.sdf
```

### 7. 评估
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/final_model.pt \
    --test_data data/test/ \
    --output eval_results.json \
    --metrics rmsd,qed,sa
```

## 目录结构

```
PharmolixFM-ablation/
├── models/                 # 模型定义
│   ├── pharmolix_fm.py    # 主模型
│   └── bfn_loss.py        # BFN 损失函数
├── data/                   # 数据加载
│   ├── molecule.py        # 分子/蛋白质/口袋定义
│   └── dataset.py         # PyTorch Dataset
├── utils/                  # 工具函数
│   ├── config.py          # 配置类
│   ├── featurizer.py      # 分子特征化
│   └── pocket_featurizer.py  # 口袋特征化
├── scripts/                # 脚本
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   ├── evaluate.py        # 评估脚本
│   └── download_data.py   # 数据下载
├── configs/                # 配置文件
│   └── pharmolix_fm.yaml
├── README.md
├── KNOWN_ISSUES.md         # 已知问题清单
└── AUDIT_REPORT.md         # 代码审计报告
```

## 使用示例

### Python API

```python
from models.pharmolix_fm import PharmolixFM
from utils.config import Config
from data.molecule import Protein, Molecule

# 创建模型
config = Config()
model = PharmolixFM(config)

# 加载检查点
checkpoint = torch.load('checkpoints/final_model.pt')
model.load_state_dict(checkpoint)

# 加载口袋和分子
protein = Protein.from_pdb_file('protein.pdb')
pocket = Pocket.from_protein_ref_ligand(protein, reference_ligand)

# 推理
molecules = model.predict_structure_based_drug_design(pocket)
```

## 已知限制

1. **预训练权重**：不支持加载 OpenBioMed 的权重，需从头训练
2. **口袋解析**：从 PDB 提取口袋原子的功能需进一步完善
3. **评估指标**：Vina 亲和力计算需安装 AutoDock Vina
4. **数据下载**：PDBbind/CrossDocked 需手动下载

详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

## 开发计划

- [x] 实现 BFN 训练损失
- [x] 实现推理脚本
- [ ] 完善评估指标（RMSD、Vina、QED、SA）
- [ ] 支持 OpenBioMed 预训练权重加载
- [ ] 添加更多数据集支持
- [ ] 分布式训练支持

## 相关项目

- [OpenBioMed](https://github.com/PharMolix/OpenBioMed) - 官方完整实现
- [MolCRAFT](https://github.com/PharMolix/OpenBioMed) - 基于 BFN 的分子生成

## 许可证

MIT License

---
**免责声明**: 本仓库为研究用途的精简版本，非 PharMolix 官方维护。训练损失函数为基于 BFN 论文的简化实现，与官方版本可能有差异。如需完整功能，请使用 [OpenBioMed](https://github.com/PharMolix/OpenBioMed)。
