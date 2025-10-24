# Jupyter Notebooks Guide

## 📚 两个Notebook概览

### 1. **experiments.ipynb** - 实验执行
用于在Google Colab或本地运行训练实验。

**功能：**
- ✅ 自动检测Colab环境并配置
- ✅ 训练单个optimizer实验
- ✅ 实时进度显示
- ✅ 自动保存结果和模型
- ✅ 生成训练曲线图

### 2. **analysis.ipynb** - 结果分析
用于分析多个实验结果并生成报告图表。

**功能：**
- ✅ 加载所有实验结果
- ✅ 统计分析（均值±标准差）
- ✅ 生成对比图
- ✅ 假设检验（H1, H2, H3）
- ✅ 收敛速度分析
- ✅ 导出LaTeX表格和CSV

---

## 🚀 使用方法

### 方法1：在Google Colab运行（推荐）

#### **运行实验：**

1. 上传 `experiments.ipynb` 到Google Colab
2. 修改第一个代码cell中的GitHub URL：
   ```python
   !git clone https://github.com/YOUR_USERNAME/optimizer_benchmark.git
   ```
3. 在"Experiment Configuration" cell中设置：
   ```python
   CONFIG = {
       'dataset': 'cifar10',
       'optimizer': 'adam',  # 更改为: sgd, adam, adamw, radam, lion
       'epochs': 200,
       'seed': 42,
       ...
   }
   ```
4. 运行所有cells（Runtime → Run all）
5. 等待训练完成（约15-20分钟/实验）
6. 结果会自动下载

#### **分析结果：**

1. 将所有实验的 `*_metrics.json` 文件放入 `results/` 目录
2. 上传 `analysis.ipynb` 到Colab
3. 运行所有cells
4. 查看对比图和统计分析

### 方法2：本地运行

```bash
# 启动Jupyter
jupyter notebook

# 打开 experiments.ipynb
# 修改CONFIG，运行所有cells

# 打开 analysis.ipynb  
# 运行所有cells查看结果
```

---

## 📊 实验工作流程

### **第1步：运行单个实验测试**
```python
# 在 experiments.ipynb 中设置
CONFIG = {
    'optimizer': 'adam',
    'dataset': 'cifar10',
    'epochs': 10,  # 先测试10个epochs
    'seed': 42
}
```
运行时间：约5分钟（测试）

### **第2步：运行完整实验（里程碑）**

对于11月6日的里程碑，你需要：

**CIFAR-10（5个优化器 × 3个seeds）：**
```python
# 运行15次，每次修改CONFIG：
experiments = [
    {'optimizer': 'sgd',   'seed': 42},
    {'optimizer': 'sgd',   'seed': 123},
    {'optimizer': 'sgd',   'seed': 456},
    {'optimizer': 'adam',  'seed': 42},
    {'optimizer': 'adam',  'seed': 123},
    {'optimizer': 'adam',  'seed': 456},
    {'optimizer': 'adamw', 'seed': 42},
    {'optimizer': 'adamw', 'seed': 123},
    {'optimizer': 'adamw', 'seed': 456},
    {'optimizer': 'radam', 'seed': 42},
    {'optimizer': 'radam', 'seed': 123},
    {'optimizer': 'radam', 'seed': 456},
    {'optimizer': 'lion',  'seed': 42},
    {'optimizer': 'lion',  'seed': 123},
    {'optimizer': 'lion',  'seed': 456},
]
```
每个实验约15-20分钟 = **总计约5小时**

**CIFAR-100（2个优化器 × 2个seeds）：**
```python
experiments = [
    {'optimizer': 'adam',  'dataset': 'cifar100', 'seed': 42},
    {'optimizer': 'adam',  'dataset': 'cifar100', 'seed': 123},
    {'optimizer': 'adamw', 'dataset': 'cifar100', 'seed': 42},
    {'optimizer': 'adamw', 'dataset': 'cifar100', 'seed': 123},
]
```
每个实验约20-25分钟 = **总计约1.5小时**

### **第3步：分析所有结果**

运行 `analysis.ipynb`，它会：
1. 自动加载所有 `results/*_metrics.json` 文件
2. 计算统计数据
3. 生成对比图
4. 测试假设
5. 导出表格

---

## 📈 输出文件

### experiments.ipynb 生成：
```
results/
├── cifar10_adam_seed42_metrics.json    # 训练指标
├── cifar10_adam_seed42_model.pth       # 模型权重
└── adam_cifar10_results.png            # 训练曲线图
```

### analysis.ipynb 生成：
```
├── CIFAR-10_comparison.png             # CIFAR-10对比图
├── CIFAR-100_comparison.png            # CIFAR-100对比图
├── H1_early_stability.png              # 假设H1分析图
├── cifar10_summary.csv                 # 统计表格
└── cifar100_summary.csv
```

---

## 🎯 假设测试

### H1: RAdam Early Training Stability
在 `analysis.ipynb` 中自动测试：
- 比较Adam和RAdam在前10个epochs的loss变化
- 计算变异系数（CV）
- 判断RAdam是否减少≥20%的方差

### H2: AdamW Regularization
在 `analysis.ipynb` 中自动测试：
- 比较Adam和AdamW的train-test gap
- 判断AdamW是否有更小的gap

### H3: Lion Robustness to Noise
需要额外实验：
```python
# 在 experiments.ipynb 中设置
CONFIG = {
    'label_noise': 0.2,  # 添加20%标签噪声
    'optimizer': 'adam'   # 然后换成'lion'
}
```
比较两者在噪声下的准确率

---

## 💡 团队协作建议

### **分工方案：**

**成员1：实验执行**
- 运行 `experiments.ipynb`
- CIFAR-10: 5个optimizers × 3 seeds
- 收集所有 `*_metrics.json` 文件

**成员2：CIFAR-100 + 假设测试**
- 运行CIFAR-100实验
- 运行H3的label noise实验
- 收集结果

**成员3：分析和可视化**
- 运行 `analysis.ipynb`
- 生成所有图表
- 准备报告内容

### **并行执行：**

如果有3个Colab账号，可以同时运行：
- Account 1: SGD + Adam实验
- Account 2: AdamW + RAdam实验  
- Account 3: Lion实验 + 分析

---

## 🔧 常见问题

### Q: Colab连接断开怎么办？
A: 代码包含checkpoint保存。不过200 epochs通常能在超时前完成。

### Q: 内存不足？
A: 减小batch_size：
```python
CONFIG['batch_size'] = 64  # 默认128
```

### Q: 如何批量运行实验？
A: 可以修改 `experiments.ipynb` 添加循环：
```python
experiments = [
    {'optimizer': 'sgd', 'seed': 42},
    {'optimizer': 'adam', 'seed': 42},
    # ...
]

for exp in experiments:
    CONFIG.update(exp)
    # 重新运行训练代码...
```

### Q: 结果文件在哪里？
A: 
- Colab: 自动下载到你的电脑
- 本地: 在 `results/` 目录

---

## 📋 检查清单

**里程碑前（11月6日）：**
- [ ] 完成CIFAR-10的15个实验
- [ ] 完成CIFAR-100的4个实验
- [ ] 运行 `analysis.ipynb` 生成对比图
- [ ] 验证所有假设测试结果
- [ ] 保存所有图表（.png文件）
- [ ] 导出统计表格（.csv文件）

**用于报告：**
- [ ] CIFAR-10对比图
- [ ] CIFAR-100对比图
- [ ] 统计表格（带均值±标准差）
- [ ] H1, H2, H3假设测试结果
- [ ] 收敛速度分析

---

## 🎓 提示

1. **先测试**：用10 epochs快速验证流程
2. **保存中间结果**：每个实验都会生成独立的文件
3. **备份数据**：下载所有 `*_metrics.json` 文件
4. **截图记录**：保存Colab中的训练曲线
5. **团队共享**：用Google Drive共享results文件夹

---

好运！🚀

