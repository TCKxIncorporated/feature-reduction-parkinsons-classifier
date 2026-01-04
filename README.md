# Sustainable Feature Reduction for Parkinson’s Detection  
### Neural Network from Scratch (NumPy)

This project explores **Parkinson’s disease detection from voice measurements** while focusing on **model sustainability**.  
Instead of blindly using all features, we reduce redundancy through **correlation-based feature selection**, then train the **same neural network from scratch** on both the full and reduced feature sets.

The goal:  
*How much model complexity can we remove before performance meaningfully drops?*

---

## Problem Overview

- Parkinson’s disease affects speech production early on  
- Voice datasets often contain **highly correlated acoustic features**
- Redundant features increase:
  - computational cost
  - training time
  - model size  
- This project balances **accuracy vs efficiency**, a key idea in sustainable ML

---

## Dataset

- **Samples:** 195  
- **Original features:** 22 acoustic voice features  
- **Target:** `status`
  - `0` → Healthy
  - `1` → Parkinson’s

The dataset is moderately imbalanced (~75% Parkinson’s).

---

## Methodology

### Data Preprocessing
- Train / validation / test split
- Feature standardization (mean = 0, std = 1)
- Stratified sampling to preserve class balance

---

### Correlation-Based Feature Reduction
- Compute feature-to-feature correlations
- If `|corr(fi, fj)| ≥ 0.85`, the pair is considered redundant
- Keep the feature with **higher correlation to the target**
- Result:
  - **22 → 12 features** (≈ 45% reduction)

---

### Neural Network (Built from Scratch)
- Implemented **entirely in NumPy**
- No TensorFlow / PyTorch
- Binary classification
- Components:
  - Forward propagation
  - Backpropagation
  - Gradient descent
  - Binary cross-entropy loss

The **same architecture** is used for both experiments — only the input size changes.

---

## Results

| Model | Features | Validation Accuracy |
|------|---------|---------------------|
| Full feature set | 22 | **78.6%** |
| Reduced feature set | 12 | **75.0%** |

### Sustainability Win
- ~42% fewer trainable parameters
- Lower memory usage
- Faster training
- Minimal accuracy loss

**Conclusion:**  
A small drop in accuracy can be a fair trade-off for large efficiency gains.

---
