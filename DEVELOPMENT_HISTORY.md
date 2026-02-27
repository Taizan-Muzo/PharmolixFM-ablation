# PharmolixFM-ablation 开发历史记录

**项目**: PharmolixFM-ablation  
**目标**: 从 OpenBioMed 提取的 PharmolixFM 精简版，用于消融实验  
**GitHub**: https://github.com/Taizan-Muzo/PharmolixFM-ablation  
**原论文**: PharmolixFM: All-Atom Molecular Foundation Model (arXiv:2503.21788)

---

## 2025-02-27 开发记录

### 初始状态
- 从 OpenBioMed 提取 PharmolixFM 代码
- 代码存在大量问题，无法运行
- 缺少 Config 类、损失函数、数据加载等

### 修复与实现

#### 1. 基础架构修复
**问题**: 代码无法导入，缺少基础类  
**修复**:
- 创建 `utils/config.py` - 简化版配置类
- 修复 `Featurized` 泛型类型 - 使用 `Generic[T]`
- 添加 `utils/pocket_featurizer.py` - 口袋特征化器

**提交**: 初始修复

#### 2. BFN 损失函数实现
**问题**: OpenBioMed 中损失函数为空（`pass`）  
**实现**:
- 创建 `models/bfn_loss.py` - 完整 BFN 损失
- `compute_sender_continuous()` - BFN 论文公式 (4)-(6)
- `compute_sender_discrete()` - BFN 论文公式 (13)-(15)
- `continuous_var_loss()` - 加权 MSE
- `discrete_var_loss()` - KL 散度

**提交**: BFN 损失实现

#### 3. 训练与推理脚本
**实现**:
- `scripts/train.py` - 完整训练循环
- `scripts/inference.py` - 对接和设计推理
- `scripts/evaluate.py` - 评估指标（RMSD、QED、SA）
- `scripts/download_data.py` - 数据下载指引
- `data/dataset.py` - PyTorch Dataset

**提交**: 可运行脚本

#### 4. Batch 并行优化
**优化**:
- 添加 `forward_pocket_molecule_docking_batch()`
- 支持任意 batch size（默认 32）
- 使用 `scatter_mean` 实现完全并行损失计算
- 消除所有 Python 循环

**提交**: True batch parallel training

---

## 当前功能状态

| 功能 | 状态 | 说明 |
|------|------|------|
| 模型架构 | ✅ 完整 | 口袋-分子对接 + 药物设计 |
| BFN 损失 | ✅ 完整 | 基于论文实现 |
| 训练 | ✅ 可用 | 支持 batch_size=32 |
| 推理 | ✅ 可用 | 对接和设计任务 |
| 评估 | ⚠️ 基础 | RMSD、QED、SA 等 |
| 预训练权重 | ❌ 不支持 | 需从头训练 |

---

## 待解决问题清单

### 🔴 高优先级
- [ ] **BFN 算法验证** - 与官方实现对比验证正确性
- [ ] **口袋原子解析** - 从 PDB 提取真实口袋原子
- [ ] **评估指标完善** - Vina 亲和力、完整 RMSD

### 🟡 中优先级
- [ ] **预训练权重支持** - 加载 OpenBioMed 检查点
- [ ] **数据加载优化** - 多进程、缓存
- [ ] **日志系统** - wandb/tensorboard

### 🟢 低优先级
- [ ] **更多数据集** - CrossDocked、ChEMBL
- [ ] **分布式训练** - DDP 支持
- [ ] **模型压缩** - 量化、剪枝

---

## 已知限制

1. **预训练权重**: 不支持加载 OpenBioMed 的权重，需从头训练
2. **口袋解析**: 从 PDB 提取口袋原子的功能需进一步完善
3. **评估指标**: Vina 亲和力计算需安装 AutoDock Vina
4. **数据下载**: PDBbind/CrossDocked 需手动下载

---

## 使用示例

```bash
# 测试安装
python -c "from models.pharmolix_fm import PharmolixFM; print('OK')"

# 虚拟数据训练
python scripts/train.py --use_dummy --epochs 5 --batch_size 32

# 推理 - 口袋分子对接
python scripts/inference.py \
    --checkpoint checkpoints/final_model.pt \
    --pdb protein.pdb --sdf ligand.sdf --task docking

# 推理 - 药物设计
python scripts/inference.py \
    --checkpoint checkpoints/final_model.pt \
    --pdb protein.pdb --task design --num_samples 10
```

---

## 技术债务

| 问题 | 影响 | 计划修复时间 |
|------|------|-------------|
| BFN 简化实现 | 可能与官方有差异 | 需验证 |
| 无单元测试 | 回归风险 | 短期 |
| 硬编码参数 | 灵活性差 | 中期 |

---

## 参考资源

- **原论文**: https://arxiv.org/abs/2503.21788
- **OpenBioMed**: https://github.com/PharMolix/OpenBioMed
- **BFN 论文**: https://arxiv.org/abs/2308.07037

---

*最后更新: 2025-02-27*
