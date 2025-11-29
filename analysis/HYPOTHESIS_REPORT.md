Based on the experimental results and the plots provided, here is a comprehensive analysis in English, formatted as a Markdown file.

***

# Optimizer Benchmark Analysis: Theoretical Interpretation of Experimental Results

## Overview
This report analyzes the performance of four distinct optimization algorithms (RAdam, AdamW, Lion, Muon) under specific constraints designed to test their theoretical properties: stability, regularization capacity, noise robustness, and data efficiency.

---

## H1: Stability Analysis (RAdam vs. Adam)

**Hypothesis:** RAdam (Rectified Adam) provides higher stability during the early phase of training compared to standard Adam by rectifying the variance of the adaptive learning rate.

### Experimental Results
* **Metric:** Coefficient of Variation (CV) of Train Loss (First 10 Epochs).
* **Outcome:** **Confirmed (✅)**.
* **Data:** RAdam-v2 reduced the CV by **10.1%** compared to Adam (0.1207 vs 0.1342).
* **Visual Evidence:** The plot shows RAdam achieving a lower and smoother loss curve immediately from Epoch 1, whereas Adam exhibits higher initial variance.

### Theoretical Interpretation
The experiment validates the core theory of **Variance Rectification**. Standard Adam struggles in the first few updates because the second moment estimate ($v_t$) is based on very few samples, leading to a divergent variance in the adaptive learning rate. This often acts like an aggressively high learning rate, destabilizing training.
RAdam introduces a rectification term $r_t$ that explicitly accounts for the degree of freedom of the variance estimator. Effectively, RAdam acts as an **automatic, dynamic warmup**, turning the adaptive learning rate off when the variance is untrustworthy and gradually enabling it as training stabilizes. The 10.1% reduction in volatility confirms that RAdam successfully mitigated the "bad start" problem without manual warmup tuning.

---

## H2: Regularization Analysis (AdamW vs. Adam)

**Hypothesis:** AdamW generalizes significantly better than Adam under high Weight Decay (WD) because it decouples weight decay from the gradient update.

### Experimental Results
* **Metric:** Test Accuracy under High Weight Decay ($1.0 \times 10^{-2}$).
* **Outcome:** **Validated (✅)**.
* **Data:** AdamW achieved **69.62%** accuracy, while Adam collapsed to **18.45%**.
* **Visual Evidence:** The bar chart shows a massive performance disparity. The "negative" train-test gap indicates strong data augmentation was used (making the training set harder than the test set), which AdamW handled effectively while Adam failed to learn.

### Theoretical Interpretation
This result provides empirical proof of the **L2 Regularization vs. Weight Decay Decoupling** theory.
* **In Adam (Coupled):** L2 regularization is added to the gradient *before* the adaptive scaling. This leads to a theoretical flaw: weights with large gradients (which are scaled down by $1/\sqrt{v_t}$) receive *less* regularization, while weights with small gradients receive *more*. This is counter-intuitive and destabilizing under high WD settings.
* **In AdamW (Decoupled):** The weight decay is applied directly to the parameters ($\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1}$), independent of the gradient scaling.
In this experiment, the high regularization pressure caused the coupled Adam optimizer to diverge or get stuck in a degenerate solution (18% accuracy), effectively preventing feature learning. AdamW, by correctly applying the decay, maintained the structural sparsity of the network without interfering with the optimization direction, leading to a healthy 69.62% accuracy.

---

## H3: Robustness Analysis (Lion vs. Adam)

**Hypothesis:** Lion (Evolved Sign Momentum) demonstrates higher robustness to label noise (30%) compared to Adam due to its sign-based update rule.

### Experimental Results
* **Metric:** Test Accuracy under 30% Label Noise.
* **Outcome:** **Rejected (❌)**.
* **Data:** Adam (**81.42%**) outperformed Lion (**75.88%**) by ~5.5%.
* **Visual Evidence:** Adam consistently achieved higher accuracy across all three random seeds.

### Theoretical Interpretation
While Lion is theoretically robust to outliers because it uses the **sign** of the gradient ($\text{sign}(m_t)$) rather than the magnitude, this experiment reveals a critical limitation: **Dependency on Batch Size**.
* **The Law of Large Numbers:** Lion relies on the sign of the mini-batch gradient to approximate the sign of the true gradient. Under 30% label noise, individual sample gradients are highly noisy/incorrect. To extract the correct "sign" (direction) from this noise, Lion requires a **large batch size** (typically $\ge 1024$) to average out the noise.
* **Adam's Advantage:** In this experiment (Batch Size = 256), the sign estimation was likely too noisy. Adam utilizes the second moment (variance, $v_t$) to normalize updates. When noise is high, the variance ($v_t$) increases, which naturally **dampens the step size** in uncertain directions. This acts as an implicit, adaptive noise filter.
**Conclusion:** At moderate batch sizes (256), Adam's variance adaptation is a more effective noise-suppression mechanism than Lion's sign-based updates.

---

## H4: Data Efficiency Analysis (Muon vs. AdamW)

**Hypothesis:** Muon achieves higher accuracy than AdamW when training data is scarce (20% of CIFAR-100) due to its orthogonalized updates.

### Experimental Results
* **Metric:** Test Accuracy on 20% Training Data.
* **Outcome:** **Validated (✅)**.
* **Data:** Muon reached **56.25%**, outperforming AdamW (**44.70%**) by a significant margin of **+11.54%**.
* **Visual Evidence:** A clear, consistent improvement across all seeds, as shown in the H4 plot.

### Theoretical Interpretation
This result highlights the power of **Structure-Aware Optimization**.
* **AdamW (Coordinate-wise):** Updates every parameter independently based on coordinate-wise statistics. On small datasets, this freedom often leads to overfitting or getting stuck in sharp minima (poor generalization).
* **Muon (Orthogonal/Spectral):** Muon constrains the update step to specific manifolds (via Newton-Schulz iteration), ensuring the weight updates are **orthogonal**. This acts as a strong, structural prior. It ensures that the gradient signal propagates efficiently through the network layers without vanishing or exploding, even when the data signal is sparse.
By forcing the optimizer to respect the spectral properties of the weight matrices, Muon extracts high-quality features from limited data much faster and more effectively than AdamW, which "wastes" data trying to determine the scale of every individual parameter.