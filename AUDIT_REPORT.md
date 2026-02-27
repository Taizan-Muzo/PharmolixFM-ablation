# PharmolixFM 代码审计报告

## 🔴 严重问题（阻止运行）

### 1. 缺少依赖包
```
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'rdkit'
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'torch_geometric'
ModuleNotFoundError: No module named 'torch_scatter'
```
**解决方案**: `pip install -r requirements.txt`

### 2. 模型未继承 LightningModule
**文件**: `scripts/inference.py`, `scripts/train.py`
**问题**: 调用 `PharmolixFM.load_from_checkpoint()` 但模型未继承 `pl.LightningModule`
**代码**:
```python
# 错误
model = PharmolixFM.load_from_checkpoint(args.checkpoint)  # 方法不存在

# 应该
model = PharmolixFM(config)
model.load_state_dict(torch.load(args.checkpoint))
```

### 3. 缺少 Config 类
**文件**: `models/pharmolix_fm.py`
**问题**: 多处使用 `self.config.xxx` 但 Config 类未定义
**解决方案**: 需要添加 Config 类或使用字典配置

### 4. Featurized 类型使用不当
**文件**: `models/pharmolix_fm.py`
**问题**: `Featurized[Molecule]` 是泛型语法，但 Featurized 是空类
**代码**:
```python
class Featurized:
    pass  # 空类，不能用作泛型

# 使用
molecule: Featurized[Molecule]  # 错误
```

---

## 🟡 中等问题（功能缺失）

### 5. 训练脚本为空实现
**文件**: `scripts/train.py`
**问题**: 只有 TODO 注释，没有实际训练逻辑

### 6. 评估脚本为空实现
**文件**: `scripts/evaluate.py`
**问题**: 只有 TODO 注释，没有实际评估逻辑

### 7. 数据下载脚本为空
**文件**: `scripts/download_data.py`
**问题**: 没有实际下载逻辑

### 8. 缺少 PocketFeaturizer 实现
**文件**: `utils/featurizer.py`
**问题**: 只有基类，没有 `PharmolixFMPocketFeaturizer` 实现

---

## 🟢 轻微问题

### 9. 类型注解警告
**文件**: `models/pharmolix_fm.py`
**问题**: `molecule: Featurized[Molecule]` 在 Python < 3.9 可能有问题

### 10. 未使用的导入
**文件**: `models/pharmolix_fm.py`
**问题**: `scatter_mean` 导入但未使用

---

## 一键运行状态

| 脚本 | 状态 | 说明 |
|------|------|------|
| `scripts/train.py` | ❌ 不可运行 | 空实现，缺少训练逻辑 |
| `scripts/inference.py` | ❌ 不可运行 | `load_from_checkpoint` 不存在 |
| `scripts/evaluate.py` | ❌ 不可运行 | 空实现 |
| `scripts/download_data.py` | ❌ 不可运行 | 空实现 |

---

## 修复建议

### 立即修复（使代码可导入）
1. 安装依赖: `pip install -r requirements.txt`
2. 添加 Config 类
3. 修复 Featurized 泛型问题

### 短期修复（使推理可运行）
4. 修改 inference.py 使用 `torch.load()` 而非 `load_from_checkpoint()`
5. 实现 PocketFeaturizer

### 长期修复（使训练可运行）
6. 实现完整训练循环
7. 继承 LightningModule 或使用标准 PyTorch 训练
