# PharmolixFM

PharMolixFM 复现版本 - 用于口袋-分子对接和基于结构的药物设计的全原子分子基础模型。

## 快速开始

### 环境设置

```bash
conda create -n pharmolixfm python=3.10 -y
conda activate pharmolixfm
pip install -r requirements.txt
```

### 训练

使用真实数据训练：
```bash
python scripts/train_full.py \
    --data_dir data/crossdocked \
    --epochs 100 \
    --batch_size 4 \
    --output_dir checkpoints/run1
```

使用虚拟数据快速测试：
```bash
python scripts/train.py \
    --use_dummy \
    --epochs 5 \
    --batch_size 8
```

### 推理

口袋-分子对接：
```bash
python scripts/inference.py \
    --checkpoint checkpoints/run1/best_model.pt \
    --pdb data/test/example_complex/protein.pdb \
    --sdf data/test/example_complex/ligand.sdf \
    --task docking \
    --output docked.sdf
```

基于结构的药物设计：
```bash
python scripts/inference.py \
    --checkpoint checkpoints/run1/best_model.pt \
    --pdb data/test/example_complex/protein.pdb \
    --task design \
    --num_samples 10 \
    --output generated.sdf
```

## 项目结构

```
.
├── models/              # 模型定义
│   ├── pharmolix_fm.py  # 主模型架构
│   └── bfn_loss.py      # BFN 损失函数
├── data/                # 数据处理
│   ├── molecule.py      # 分子/蛋白质数据类
│   ├── dataset.py       # PyTorch Dataset
│   └── crossdocked_dataset.py  # CrossDocked 数据集
├── utils/               # 工具函数
│   ├── config.py        # 配置类
│   ├── featurizer.py    # 分子特征化
│   └── pocket_featurizer.py  # 口袋特征化
├── scripts/             # 训练和推理脚本
│   ├── train.py         # 基础训练（支持虚拟数据）
│   ├── train_full.py    # 完整训练
│   ├── train_ddp.py     # 多 GPU 训练
│   ├── inference.py     # 推理脚本
│   └── evaluate.py      # 评估脚本
├── configs/             # 配置文件
├── data/crossdocked/    # CrossDocked2020 数据集（符号链接）
└── requirements.txt     # 依赖列表
```

## 数据

CrossDocked2020 数据已放在 `/run/ti/pharmolixfm_data/crossdocked/`，通过符号链接 `data/crossdocked` 访问。

## 引用

```bibtex
@article{luo2025pharmolixfm,
  title={PharMolixFM: All-Atom Foundation Models for Molecular Modeling and Generation},
  author={Luo, Yizhen and Wang, Jiashuo and Fan, Siqi and Nie, Zaiqing},
  journal={arXiv preprint arXiv:2503.21788},
  year={2025}
}
```
