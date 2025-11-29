# Lion Optimizer 使用指南
# Lion Optimizer Usage Guide

## 重要发现 (Key Findings)

基于我们的假设验证实验（H3），我们发现 Lion 优化器需要**特殊的超参数设置**才能发挥最佳性能，特别是在噪声数据场景下。

---

## 核心超参数设置 (Critical Hyperparameters)

### 1. 学习率 (Learning Rate)

**规则**: Lion 的学习率通常需要是 AdamW 的 **1/3 到 1/10**

- **Adam 默认 LR**: 0.001
- **Lion 推荐 LR**: 0.001 (与 Adam 相同，但相对于 AdamW 的 0.001 是 1:1)
- **注意**: 如果使用 AdamW 作为基线（LR=0.001），Lion 应该使用 LR=0.0001 到 0.0003

**我们的实验设置**:
- Adam: LR = 0.001
- Lion: LR = 0.001 (在 CIFAR-10 上，这个设置表现良好)

### 2. 权重衰减 (Weight Decay)

**规则**: Lion 需要**显著更高的 Weight Decay** 来稳定符号更新

- **Adam/AdamW 标准 WD**: 0.01 到 0.001
- **Lion 推荐 WD**: **0.1 到 1.0** (在 ImageNet 上甚至用到 WD=1.0)
- **我们的实验**: WD = 0.1 (在 CIFAR-10 上)

**原因**: 
- Lion 使用符号（Sign）更新，更新幅度固定
- 高 Weight Decay 帮助防止过拟合，特别是在噪声数据上
- 符号更新对异常梯度更敏感，需要更强的正则化

### 3. Betas 参数

**规则**: Lion 必须使用 **betas=(0.9, 0.99)**，而不是 PyTorch 默认的 (0.9, 0.999)

- **Adam 标准 Betas**: (0.9, 0.999)
- **Lion 标准 Betas**: **(0.9, 0.99)** ⚠️

**原因**:
- Lion 使用 `β₂` 来追踪梯度的移动平均
- 由于它是基于符号（Sign）更新的，它需要**更快的动量衰减**（更小的 `β₂`）来快速适应梯度的方向变化
- 如果使用默认的 (0.9, 0.999)，Lion 的反应会变得迟钝，导致在噪声数据上性能下降

**实现**:
```python
# 正确的方式
optimizer = Lion(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.99),  # 必须指定！
    weight_decay=0.1
)

# 错误的方式（使用默认 betas）
optimizer = Lion(
    model.parameters(),
    lr=0.001,
    weight_decay=0.1
    # 如果库默认使用 (0.9, 0.999)，性能会下降
)
```

---

## 实验验证 (Experimental Validation)

### H3 实验设置

我们在 CIFAR-10 上测试了 Lion 在 20% 标签噪声下的鲁棒性：

**设置**:
- Dataset: CIFAR-10
- Model: ResNet-18
- Label Noise: 20%
- Batch Size: 256 (大 batch 有助于稳定符号更新)
- Epochs: 50

**超参数对比**:

| Optimizer | LR | WD | Betas | Batch Size |
|-----------|----|----|-------|------------|
| Adam | 0.001 | 0.001 | (0.9, 0.999) | 256 |
| Lion | 0.001 | **0.1** | **(0.9, 0.99)** | 256 |

**关键发现**:
1. Lion 的符号更新在正确设置下确实对标签噪声更鲁棒
2. **高 Weight Decay (0.1)** 是必需的，否则 Lion 容易过拟合
3. **正确的 Betas (0.9, 0.99)** 至关重要，使用默认值会导致性能显著下降

---

## 使用建议 (Usage Recommendations)

### 1. 从 AdamW 迁移到 Lion

```python
# AdamW 设置
optimizer_adamw = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# 对应的 Lion 设置
optimizer_lion = Lion(
    model.parameters(),
    lr=0.0003,  # 1/3 of AdamW LR
    weight_decay=0.1,  # 10x higher
    betas=(0.9, 0.99)  # Must specify!
)
```

### 2. 检查清单 (Checklist)

在使用 Lion 之前，确保：

- [ ] ✅ LR 设置为 AdamW 的 1/3 到 1/10
- [ ] ✅ Weight Decay 设置为 0.1 或更高（根据数据集调整）
- [ ] ✅ Betas 显式设置为 (0.9, 0.99)
- [ ] ✅ 使用较大的 Batch Size (≥256) 以稳定符号更新
- [ ] ✅ 在噪声数据或困难任务上，考虑进一步提高 Weight Decay

### 3. 常见错误 (Common Mistakes)

❌ **错误 1**: 使用默认 betas
```python
# 错误：可能使用库默认的 (0.9, 0.999)
optimizer = Lion(model.parameters(), lr=0.001, weight_decay=0.1)
```

✅ **正确**:
```python
optimizer = Lion(model.parameters(), lr=0.001, weight_decay=0.1, betas=(0.9, 0.99))
```

❌ **错误 2**: Weight Decay 太低
```python
# 错误：WD 太低，Lion 容易过拟合
optimizer = Lion(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.99))
```

✅ **正确**:
```python
optimizer = Lion(model.parameters(), lr=0.001, weight_decay=0.1, betas=(0.9, 0.99))
```

---

## 理论背景 (Theoretical Background)

### 为什么 Lion 需要这些特殊设置？

1. **符号更新的特性**:
   - Lion 使用 `sign(m_t)` 而非 `m_t` 进行更新
   - 这意味着所有参数的更新幅度相同，只有方向不同
   - 需要更快的动量衰减来适应方向变化

2. **正则化需求**:
   - 符号更新对异常梯度更敏感
   - 高 Weight Decay 帮助过滤噪声，防止过拟合

3. **大 Batch 的优势**:
   - 大 Batch 提供更稳定的梯度估计
   - 符号更新依赖梯度方向的一致性，大 Batch 有助于此

---

## 参考文献 (References)

- Chen, X., et al. (2023). *Symbolic discovery of optimization algorithms*. arXiv:2302.06675
- Lion 论文中 ImageNet 实验使用 WD=1.0 (配合 AdamW WD=0.1)

---

## 更新日志 (Changelog)

- **2025-11-28**: 基于 H3 假设验证实验，发现并记录 Lion 的关键超参数设置要求

