# 数据增强控制指南
# Data Augmentation Control Guide

本文档说明如何在假设测试中控制数据增强的强度。

---

## 数据增强级别 (Augmentation Levels)

### CIFAR-10 (使用 `src/train.py`)

**控制方式**: `--augment` / `--no-augment` 参数

- **`--augment`** (默认): 基础数据增强
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip
  
- **`--no-augment`**: 无数据增强（仅归一化）

**示例**:
```bash
# 使用数据增强（默认）
python src/train.py --dataset cifar10 --optimizer adam --augment

# 关闭数据增强
python src/train.py --dataset cifar10 --optimizer adam --no-augment
```

### CIFAR-100 (使用 `train_cifar100_optimized.py`)

**控制方式**: `--augmentation` 参数，可选值：`basic`, `medium`, `strong`

- **`basic`**: 基础增强
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip
  
- **`medium`**: 中等增强
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip
  - ColorJitter (brightness, contrast, saturation, hue)
  - RandomRotation(15°)
  
- **`strong`** (默认): 强增强
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip
  - AutoAugment (CIFAR-10 policy)
  - Cutout (16×16 holes)

**示例**:
```bash
# 使用强数据增强（默认）
python train_cifar100_optimized.py --optimizer adamw --augmentation strong

# 使用中等数据增强
python train_cifar100_optimized.py --optimizer adamw --augmentation medium

# 使用基础数据增强
python train_cifar100_optimized.py --optimizer adamw --augmentation basic
```

---

## 假设测试中的数据增强设置

### H1: RAdam 稳定性
- **设置**: 使用默认（CIFAR-100 使用 `strong`）
- **原因**: 关注早期训练稳定性，数据增强不影响方差修正机制

### H2: AdamW 正则化
- **当前设置**: `--augmentation medium`
- **原因**: 
  - 减少强数据增强导致的 Train Acc < Test Acc 现象
  - 更清楚地观察 AdamW vs Adam 在正则化上的差异
  - 如果仍想看到更明显的差异，可以改为 `basic` 或完全关闭

**修改建议**:
```bash
# 如果想完全关闭数据增强来测试（需要修改 train_cifar100_optimized.py）
# 或者使用 basic 级别
--augmentation basic
```

### H3: Lion 鲁棒性
- **当前设置**: `--augment` (基础增强)
- **原因**: 
  - 关注标签噪声的鲁棒性，不需要过强的数据增强
  - 基础增强足以提供必要的正则化

### H4: Muon 数据效率
- **当前设置**: 使用默认（`strong`）
- **原因**: 
  - 在有限数据下，数据增强有助于提高数据利用效率
  - 可以测试 Muon 在强增强下的表现

---

## 如何修改假设测试中的数据增强

### 方法 1: 直接修改 `run_hypothesis_tests.py`

在相应的假设测试部分添加或修改 `--augmentation` 参数：

```python
# H2 示例：使用 basic 增强
cmd = [
    sys.executable, 'train_cifar100_optimized.py',
    '--optimizer', 'adam',
    '--augmentation', 'basic',  # 修改这里
    # ... 其他参数
]

# H3 示例：关闭数据增强
cmd = [
    sys.executable, 'src/train.py',
    '--dataset', 'cifar10',
    '--optimizer', 'adam',
    '--no-augment',  # 添加这个参数
    # ... 其他参数
]
```

### 方法 2: 添加命令行参数（高级）

可以在 `run_hypothesis_tests.py` 中添加全局的数据增强控制参数：

```python
parser.add_argument('--aug-level', type=str, default=None,
                    choices=['none', 'basic', 'medium', 'strong'],
                    help='Override augmentation level for all experiments')
```

---

## 数据增强对实验结果的影响

### 强数据增强的影响

1. **Train Accuracy < Test Accuracy**:
   - 强增强（如 AutoAugment + Cutout）使训练数据"变难"
   - 测试时没有增强，所以 Test Acc 可能更高
   - 这是**正常现象**，不代表过拟合

2. **对 H2 的影响**:
   - 强增强会"掩盖" AdamW vs Adam 在正则化上的差异
   - 建议使用 `medium` 或 `basic` 来更清楚地观察差异

3. **对 H4 的影响**:
   - 在有限数据（20%）下，数据增强是必要的
   - 但可以测试不同增强级别下 Muon 的优势

---

## 建议的数据增强设置

| 假设 | 推荐设置 | 原因 |
|------|---------|------|
| H1 | `strong` (默认) | 不影响稳定性分析 |
| H2 | `medium` 或 `basic` | 更清楚地观察正则化差异 |
| H3 | `basic` 或 `--augment` | 关注噪声鲁棒性，不需要过强增强 |
| H4 | `strong` (默认) | 有限数据下需要增强 |

---

## 快速修改脚本

如果需要快速修改所有假设测试的数据增强设置，可以在 `run_hypothesis_tests.py` 的开头定义：

```python
# 全局数据增强设置
AUG_LEVEL_H1 = 'strong'  # 或 'medium', 'basic'
AUG_LEVEL_H2 = 'medium'  # 推荐 medium 或 basic
AUG_LEVEL_H3 = 'basic'   # 或使用 --augment
AUG_LEVEL_H4 = 'strong'  # 或 'medium'
```

然后在各个假设测试中使用这些变量。

