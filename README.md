# PharmolixFM

PharmolixFM 独立运行版本 - 全原子分子基础模型

## 论文
[PharmolixFM: All-Atom Molecular Foundation Model](https://arxiv.org/abs/2503.21788)

## 功能
- Pocket-Molecule Docking（口袋-分子对接）
- Structure-Based Drug Design（基于结构的药物设计）

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备
```bash
python scripts/download_data.py
```

### 2. 训练
```bash
python scripts/train.py --config configs/pharmolix_fm.yaml
```

### 3. 评估
```bash
python scripts/evaluate.py --checkpoint checkpoints/pharmolix_fm.ckpt
```

### 4. 推理
```bash
python scripts/inference.py --pdb protein.pdb --sdf ligand.sdf
```

## 目录结构
```
PharmolixFM/
├── models/           # 模型定义
├── data/             # 数据加载器
├── configs/          # 配置文件
├── scripts/          # 训练和评估脚本
├── utils/            # 工具函数
└── tests/            # 单元测试
```

## 许可证
MIT
