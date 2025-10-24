# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Test Your Setup (1 minute)

```bash
# Run quick test
python test_setup.py
```

Expected output:
```
============================================================
Testing Optimizer Benchmark Setup
============================================================

✓ PyTorch version: 2.x.x
✓ CUDA available: True/False
✓ Testing data loading (CIFAR-10)...
✓ Testing model (ResNet-18)...
✓ Testing optimizers...
✓ Running quick training test...

✓ All tests passed! Setup is working correctly.
============================================================
```

### Step 3: Run Your First Experiment (15-20 minutes)

```bash
# Train ResNet-18 on CIFAR-10 with Adam optimizer
python run_single_experiment.py --optimizer adam --dataset cifar10 --seed 42
```

This will:
- Download CIFAR-10 dataset automatically
- Train for 200 epochs (~15-20 min on GPU)
- Save results to `results/cifar10_adam_lr0.001_wd0.0_seed42/`

### Step 4: Visualize Results

```bash
# After running a few experiments, generate plots
python visualize_results.py
```

## 📊 Running Your First Comparison

Compare 2-3 optimizers quickly:

```bash
# Run Adam (15-20 min)
python run_single_experiment.py --optimizer adam --dataset cifar10 --seed 42

# Run AdamW (15-20 min)
python run_single_experiment.py --optimizer adamw --dataset cifar10 --seed 42

# Run RAdam (15-20 min)
python run_single_experiment.py --optimizer radam --dataset cifar10 --seed 42

# Visualize comparison
python visualize_results.py
```

Total time: ~1 hour on GPU

## 🎯 For Your Milestone (Nov 6)

### CIFAR-10 Complete Experiments (5 optimizers × 3 seeds)

```bash
# This will take ~5 hours on GPU
python run_all_optimizers.py --datasets cifar10 --epochs 200
```

### CIFAR-100 Partial Results (2 optimizers × 2 seeds)

```bash
# Adam experiments (~50 min)
python run_single_experiment.py --optimizer adam --dataset cifar100 --seed 42
python run_single_experiment.py --optimizer adam --dataset cifar100 --seed 123

# AdamW experiments (~50 min)
python run_single_experiment.py --optimizer adamw --dataset cifar100 --seed 42
python run_single_experiment.py --optimizer adamw --dataset cifar100 --seed 123
```

Total time: ~7 hours on GPU

## 💡 Tips for Google Colab

```python
# 1. Upload your code
from google.colab import files
!git clone <your-repo-url>
%cd optimizer_benchmark

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Check GPU
!nvidia-smi

# 4. Run experiment
!python run_single_experiment.py --optimizer adam --dataset cifar10 --seed 42

# 5. Download results
!zip -r results.zip results/
files.download('results.zip')
```

## 🔬 Testing Your Hypotheses

### H1: RAdam Early Training Stability

```bash
# Run on CIFAR-100 to see the difference
python src/train.py --dataset cifar100 --optimizer adam --seed 42
python src/train.py --dataset cifar100 --optimizer radam --seed 42

# Check first 10 epochs in results/*/metrics.json
```

### H2: AdamW Regularization

```bash
# Compare with high weight decay
python src/train.py --dataset cifar10 --optimizer adam --weight-decay 0.001 --seed 42
python src/train.py --dataset cifar10 --optimizer adamw --weight-decay 0.001 --seed 42
```

### H3: Lion Robustness to Label Noise

```bash
# Add 20% label noise
python src/train.py --dataset cifar10 --optimizer adam --label-noise 0.2 --seed 42
python src/train.py --dataset cifar10 --optimizer lion --label-noise 0.2 --seed 42
```

## 📈 Understanding Results

After training, check these files:

```
results/cifar10_adam_lr0.001_wd0.0_seed42/
├── config.yaml         # Your experiment settings
├── metrics.json        # All metrics per epoch
├── summary.yaml        # Final results
├── best_model.pth      # Best checkpoint
└── final_model.pth     # Final checkpoint
```

Key metrics in `metrics.json`:
- `test_acc`: Test accuracy per epoch
- `train_loss`: Training loss per epoch
- `learning_rate`: LR schedule

## 🐛 Common Issues

### Out of Memory
```bash
# Reduce batch size
python src/train.py --batch-size 64 ...
```

### Slow Training
```bash
# Check if using GPU
python -c "import torch; print(torch.cuda.is_available())"

# Reduce workers if CPU bottleneck
python src/train.py --workers 2 ...
```

### Lion Not Available
```bash
pip install lion-pytorch
```

## 📝 Next Steps

1. ✅ Run test_setup.py to verify installation
2. ✅ Run your first experiment (adam on cifar10)
3. ✅ Compare 2-3 optimizers
4. ✅ Start milestone experiments
5. ✅ Generate plots with visualize_results.py
6. ✅ Analyze results for your report

## 🆘 Need Help?

- Check README.md for detailed documentation
- Look at configs/example_config.yaml for all options
- Run `python src/train.py --help` for all arguments

Good luck with your experiments! 🎉

