# CSE 493S Project Proposal  
### Systematic Comparison of Adam-based Optimization Algorithms with Theoretical Analysis  

**Authors:**  
- Jinghao Liu (jliu63)  
- Xuan Zhang (xuanz24)  
- Yuzheng Zhang (yuzhez4)  

**Date:** October 17, 2025  

---

## Team Information

- **Student 1:** Jinghao Liu, NetID: jliu63, Status: Undergraduate  
- **Student 2:** Xuan Zhang, NetID: xuanz24, Status: Undergraduate  
- **Student 3:** Yuzheng Zhang, NetID: yuzhez4, Status: Undergraduate  

---

## Project Direction

This project follows the **fair comparisons for algorithms** direction with a theoretical component.  
We will conduct a systematic empirical comparison of five optimization algorithms (SGD with momentum, Adam, AdamW, RAdam, and Lion) while analyzing their mathematical foundations to understand when and why certain optimizers perform better in different scenarios.

---

## Citations

We will base our comparison on the following papers:

1. **SGD with Momentum:** Sutskever, I., et al. (2013). *On the importance of initialization and momentum in deep learning.* *ICML.* [Link](http://proceedings.mlr.press/v28/sutskever13.pdf)
2. **Adam:** Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
3. **AdamW:** Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* *ICLR.* [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
4. **RAdam:** Liu, L., et al. (2020). *On the variance of the adaptive learning rate and beyond.* *ICLR.* [arXiv:1908.03265](https://arxiv.org/abs/1908.03265)
5. **Lion:** Chen, X., et al. (2023). *Symbolic discovery of optimization algorithms.* [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)

---

## Goals and Expected Outcomes

### Primary Objective

Provide a comprehensive comparison of five stable optimization algorithms (SGD with momentum, Adam, AdamW, RAdam, and Lion) combining empirical evaluation with theoretical analysis to understand their mathematical foundations and practical performance.

### Concrete Deliverables

1. **Performance comparison:** Training loss and validation accuracy curves for all five optimizers on CIFAR-10 and CIFAR-100, averaged across multiple seeds  
2. **Quantitative metrics:** Performance table showing final test accuracy, convergence speed (epochs to target accuracy), and training stability (standard deviation across seeds)  
3. **Theoretical analysis:** Mathematical derivation and comparison of update rules, identifying key algorithmic differences  
4. **Theory-driven experiments:** Validation of 3 theoretical hypotheses through specifically designed experiments (e.g., early training stability for RAdam, noise robustness for Lion, regularization effectiveness for AdamW)  
5. **Hyperparameter sensitivity:** Analysis of optimizer performance across different learning rates  
6. **Public GitHub repository:** Complete code with documentation for reproducibility  
7. **Final report:** 8-page technical report integrating theoretical insights with empirical findings  

---

## First Milestone (by Nov 6)

By **November 6**, we will complete:

1. **Theoretical framework:** Document containing update rule derivations for all five optimizers and 3 testable hypotheses with mathematical justification  
2. **Complete CIFAR-10 experiments:** All 5 optimizers × 3 seeds = 15 experiments with comparison plots and performance table  
3. **Partial CIFAR-100 results:** At least 2 optimizers with 2 seeds each to demonstrate computational feasibility  
4. **Working pipeline:** Unified training framework with automated logging and visualization scripts  
5. **Preliminary analysis:** Initial validation of at least one theoretical hypothesis using CIFAR-10 results  

This milestone demonstrates that our experimental setup is sound and validates the feasibility of completing both empirical and theoretical components.

---

## Empirical Project Details

### Data Access

Both CIFAR-10 and CIFAR-100 are publicly available through PyTorch's `torchvision.datasets` module and will be automatically downloaded. No special permissions required.

### Dataset and Model Sizes

- **CIFAR-10:** 50,000 training images (32×32 RGB, 10 classes), 10,000 test images.  
  Model: ResNet-18 with ~11M parameters  
- **CIFAR-100:** 50,000 training images (32×32 RGB, 100 classes), 10,000 test images.  
  Model: ResNet-18 with ~11M parameters (adjusted final layer for 100 classes)  

### Code Implementation

We will use existing implementations where available:

- **SGD with Momentum, Adam, AdamW, RAdam:** PyTorch built-in (`torch.optim`)  
- **Lion:** [`lion-pytorch`](https://github.com/lucidrains/lion-pytorch)

Note: Prodigy is excluded due to implementation stability concerns identified during preliminary testing.  

We will implement our own training loop to ensure fair comparison with identical hyperparameters, data augmentation, and evaluation protocols.

### Computational Feasibility

**Resource requirements:**

- CIFAR-10: 15–20 minutes per experiment (Colab T4 GPU), total ≈5 hours for 15 experiments (5 optimizers × 3 seeds)  
- CIFAR-100: 20–25 minutes per experiment (Colab T4 GPU), total ≈6 hours for 15 experiments  
- Main experiments: 5 optimizers × 2 datasets × 3 seeds = 30 experiments, ≈11 total GPU hours  
- Hyperparameter sensitivity: Additional 3 learning rates × 3 key optimizers × 1 dataset = 9 experiments, ≈3 GPU hours  
- Theory-driven hypothesis testing: 3 specific experiments, ≈1.5 GPU hours  

**Available resources:**  
Colab free tier (T4 GPU) for all experiments. We have validated that ResNet-18 trains on CIFAR-10 in under 20 minutes per run on a T4 GPU with sufficient memory headroom. CIFAR-100 requires slightly longer training but remains feasible.

**Feasibility argument:**  
Total GPU time of ~15.5 hours is well within Colab's free quota when distributed across team members (3 members × 5 hours each) and multiple sessions. Each team member can contribute 1-2 hours of GPU time per day over 7-10 days.

---

## Theoretical Component

### Mathematical Analysis

We will formally derive and compare the update rules for all five optimizers:

- **Moment estimation:** How optimizers accumulate gradient information (exponential moving average vs sign-based)  
- **Adaptive rates:** Per-parameter learning rate computation mechanisms  
- **Weight decay:** L2 penalty (Adam) vs decoupled weight decay (AdamW)  
- **Variance control:** Mathematical analysis of RAdam's rectification term  
- **Sign-based updates:** Lion's use of `sign(m_t)` for robustness to noise
- **Comparison with SGD+momentum:** Understanding when simple momentum-based optimization suffices  

### Theory-Driven Hypotheses

We will formulate and empirically test the following hypotheses:

1. **H1 (RAdam stability):** RAdam's variance rectification will reduce coefficient of variation in loss by at least 20% during the first 10 epochs compared to Adam, particularly evident on CIFAR-100.  
2. **H2 (AdamW regularization):** AdamW will show a smaller train-test accuracy gap than Adam at high weight decay values (≥10⁻³), demonstrating better generalization through decoupled weight decay.  
3. **H3 (Lion robustness):** Lion will maintain higher relative accuracy than Adam under 20% label noise on CIFAR-10 due to sign-based updates that naturally filter gradient noise.  

Each hypothesis will be validated through specifically designed experiments with quantitative metrics.

### Expected Theoretical Contributions

- Quantitative validation or refutation of theoretical predictions  
- Analysis of theory–practice gaps and their implications  
- Insights into when mathematical properties translate to empirical advantages  

---

## Timeline and Risk Mitigation

**Timeline:**  
- **Week 1 (Oct 20-26):** Framework implementation + CIFAR-10 baseline + theory formulation  
- **Week 2 (Oct 27-Nov 2):** Complete CIFAR-10 experiments for all 5 optimizers  
- **Week 3 (Nov 3-9):** CIFAR-100 experiments + hypothesis testing (Milestone due Nov 6)  
- **Week 4 (Nov 10-16):** Hyperparameter sensitivity analysis + additional experiments  
- **Week 5-6 (Nov 17-30):** Data analysis + report writing + final revisions  

**Risk mitigation:**  
- If GPU quota is limiting, team members can use separate Colab accounts (3 members = 3× quota), and CIFAR-10 experiments can serve as primary results with CIFAR-100 as extended validation.  
- If CIFAR-100 proves too computationally expensive, we can conduct more in-depth analysis on CIFAR-10 with various training conditions (label noise, limited data, different architectures).  
- If Lion implementation has issues, we still have 4 well-established optimizers (SGD+momentum, Adam, AdamW, RAdam) for solid comparison.  

---