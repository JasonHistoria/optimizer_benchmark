# 假设验证实验指南
# Hypothesis Validation Guide

本文档说明如何运行所有假设验证实验。

**重要**: 假设验证实验的结果保存在独立的目录 `results_hypothesis/` 中，与主实验结果 `results/` 分离，避免混淆。

---

## 实验概览

### H1: RAdam 早期训练稳定性 ✅ (已部分验证)
**假设**: RAdam 的方差修正机制将在前 10 个 epoch 将损失变异系数降低至少 20%（相比 Adam）

**状态**: 已通过 RAdam-v2 vs Adam 的对比验证，但可以收集更精确的数据

### H2: AdamW 正则化优势 ⚠️ (部分验证)
**假设**: AdamW 在高权重衰减（≥10⁻³）下将展现比 Adam 更小的 train-test gap（10-15%）

**状态**: 已部分验证（差异约 5.5%），需要更高 weight_decay 的实验

### H3: Lion 鲁棒性 ❌ (待验证)
**假设**: Lion 在 20% 标签噪声下将保持比 Adam 更高的相对准确率

**状态**: 需要运行专门的标签噪声实验

### H4: Muon 数据效率 ⚠️ (待验证)
**假设**: Muon 的正交化更新机制使其在有限训练数据（20%）下表现更好，预期比 AdamW 高 5-10% 准确率

**状态**: 需要运行有限数据实验

---

## 命名规则

假设验证实验的结果保存在 `results_hypothesis/` 目录，命名规则如下：

- **H1**: `cifar100_wrn-16-4_{optimizer}_h1_seed{seed}/`
- **H2**: `cifar100_wrn-16-4_{optimizer}_h2_wd0.01_seed{seed}/`
- **H3**: `cifar10_{optimizer}_lr*_wd*_seed{seed}/` (config.yaml 中 label_noise=0.2)
- **H4**: `cifar100_wrn-16-4_{optimizer}_h4_data20pct_seed{seed}/`

---

## 运行实验

### 方法 1: 使用自动化脚本（推荐）

```bash
# 运行所有假设验证实验
python3.9 run_hypothesis_tests.py --all

# 或分别运行
python3.9 run_hypothesis_tests.py --h1  # RAdam 稳定性
python3.9 run_hypothesis_tests.py --h2  # AdamW 正则化
python3.9 run_hypothesis_tests.py --h3  # Lion 鲁棒性
python3.9 run_hypothesis_tests.py --h4  # Muon 数据效率
```

### 方法 2: 手动运行（用于调试或单次实验）

#### H1: RAdam 稳定性测试

```bash
# 运行 Adam 基线（用于对比）
python3.9 train_cifar100_optimized.py \
  --optimizer adam \
  --epochs 100 \
  --seed 42 \
  --batch-size 128 \
  --model wrn-16-4 \
  --scheduler cosine \
  --save-dir ./results_hypothesis \
  --exp-suffix _h1

# RAdam-v2 已经存在，可以直接分析
# 结果目录: results/cifar100_wrn-16-4_radam-v2_seed{42,123,456}/
```

**分析**: 比较前 10 个 epoch 的 loss 变异系数（CV = std/mean）

#### H2: AdamW 正则化测试

```bash
# Adam with high weight_decay=0.01
python3.9 train_cifar100_optimized.py \
  --optimizer adam \
  --epochs 100 \
  --seed 42 \
  --batch-size 128 \
  --model wrn-16-4 \
  --scheduler cosine \
  --weight-decay 0.01 \
  --save-dir ./results_hypothesis \
  --exp-suffix _h2_wd0.01

# AdamW with high weight_decay=0.01
python3.9 train_cifar100_optimized.py \
  --optimizer adamw \
  --epochs 100 \
  --seed 42 \
  --batch-size 128 \
  --model wrn-16-4 \
  --scheduler cosine \
  --weight-decay 0.01 \
  --save-dir ./results_hypothesis \
  --exp-suffix _h2_wd0.01

# 运行 3 个种子 (42, 123, 456)
```

**分析**: 比较 train-test accuracy gap

#### H3: Lion 鲁棒性测试（20% 标签噪声）

```bash
# Adam with 20% label noise
python3.9 src/train.py \
  --dataset cifar10 \
  --optimizer adam \
  --epochs 100 \
  --seed 42 \
  --batch-size 128 \
  --model resnet18 \
  --scheduler cosine \
  --label-noise 0.2 \
  --save-dir ./results_hypothesis

# Lion with 20% label noise
python3.9 src/train.py \
  --dataset cifar10 \
  --optimizer lion \
  --epochs 100 \
  --seed 42 \
  --batch-size 128 \
  --model resnet18 \
  --scheduler cosine \
  --label-noise 0.2 \
  --save-dir ./results_hypothesis

# 运行 3 个种子 (42, 123, 456)
```

**分析**: 比较最终准确率和相对准确率下降

#### H4: Muon 数据效率测试（20% 训练数据）

```bash
# AdamW with 20% training data
python3.9 train_cifar100_optimized.py \
  --optimizer adamw \
  --epochs 100 \
  --seed 42 \
  --batch-size 128 \
  --model wrn-16-4 \
  --scheduler cosine \
  --data-fraction 0.2 \
  --save-dir ./results_hypothesis \
  --exp-suffix _h4_data20pct

# Muon with 20% training data
python3.9 train_cifar100_optimized.py \
  --optimizer muon \
  --epochs 100 \
  --seed 42 \
  --batch-size 128 \
  --model wrn-16-4 \
  --scheduler cosine \
  --data-fraction 0.2 \
  --save-dir ./results_hypothesis \
  --exp-suffix _h4_data20pct

# 运行 3 个种子 (42, 123, 456)
```

**分析指标**:
1. **最终准确率**: Muon 应该比 AdamW 高 5-10%
2. **收敛速度**: Muon 可能更快达到较高准确率
3. **数据利用效率**: Muon 的正交化更新应该能更好地利用有限数据

**预期结果**：
- Muon 在 20% 数据下可能达到 65-68% 准确率，而 AdamW 可能只有 60-63%
- 这展现了 Muon 在数据稀缺场景下的优势

---

## 在 Hyak 上运行

### 提交 SLURM 任务

```bash
# 创建假设验证脚本
cat > submit_hypothesis_hyak.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=hyp_test
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hypothesis-%j.out
#SBATCH --error=logs/hypothesis-%j.err

cd /mmfs1/home/jliu63/project/optimizer_benchmark

# Load modules
module load python3/3.10 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true

# Activate environment
source venv/bin/activate

# Run hypothesis tests
python3.9 run_hypothesis_tests.py --h3  # 或 --h4, --all
EOF

chmod +x submit_hypothesis_hyak.sh
sbatch submit_hypothesis_hyak.sh
```

### 或使用交互式会话

```bash
# 申请交互式 GPU 节点
salloc -A rselab -p ckpt --gres=gpu:a40:1 -c 4 --mem=32G --time=4:00:00

# 在计算节点上运行
cd /mmfs1/home/jliu63/project/optimizer_benchmark
source venv/bin/activate
python3.9 run_hypothesis_tests.py --h3  # 或 --h4
```

---

## 实验时间估算

- **H1**: 已基本完成，只需验证数据（~0 小时）
- **H2**: 6 个实验（Adam/AdamW × 3 seeds），每个 ~1.2 小时 = **~7.2 小时**
- **H3**: 6 个实验（Adam/Lion × 3 seeds），每个 ~0.3 小时 = **~1.8 小时**
- **H4**: 6 个实验（AdamW/Muon × 3 seeds），每个 ~1.2 小时 = **~7.2 小时**

**总计**: 约 **16.2 小时 GPU 时间**

---

## 结果分析

实验完成后，结果将保存在 `results_hypothesis/` 目录。

### 使用分析脚本

```bash
# 分析所有假设
python3.9 analyze_hypothesis_results.py --all

# 或分别分析
python3.9 analyze_hypothesis_results.py --h1
python3.9 analyze_hypothesis_results.py --h2
python3.9 analyze_hypothesis_results.py --h3
python3.9 analyze_hypothesis_results.py --h4
```

分析结果将输出到控制台，并显示每个假设的验证状态。

---

## 更新报告

实验完成后，更新 `analysis/COMPREHENSIVE_REPORT.md` 中的假设验证章节（第 6 节）。

---

## 注意事项

1. **结果目录分离**: 假设验证结果保存在 `results_hypothesis/`，不会与主实验结果混淆
2. **H3 实验**: 标签噪声实验的结果目录名可能不包含 `label_noise` 标识，需要检查 `config.yaml` 确认
3. **H2 实验**: 高 weight_decay 可能导致训练不稳定，注意监控 loss
4. **H4 实验**: 有限数据实验使用固定的随机种子（42）来选择数据子集，确保可重复性
5. **资源管理**: 如果 GPU 资源有限，可以分批运行（建议优先级：H3 > H4 > H2）
