# HT2P-Hyperparameter-Tuned-Two-stage-Pipeline-for-Imbalanced-Tabular-Data

This repository presents HT2P, a novel and advanced two‑stage learning pipeline designed for imbalanced tabular data classification. The framework builds upon TabNet and enhances it with self‑supervised pretraining, feature engineering, and hyperparameter tuning to significantly improve predictive performance and model robustness.

---

##  Motivation

Although TabNet offers interpretable and powerful learning for tabular data through sequential attention, its performance degrades under class imbalance particularly in recall and F1-score.  
Traditional remedies such as class weighting or resampling are often insufficient and may introduce instability or overfitting.

HT2P addresses this limitation through:
- Systematic hyperparameter optimisation
- Representation learning via self-supervised pretraining
- Robust inference strategies tailored for imbalanced classification

---

##  Method Overview

HT2P follows a **two-stage pipeline**:

### Stage I – Hyperparameter-Tuned Architecture Search
A large-scale grid–random search explores more than 10,000 TabNet configurations, optimising:
- Decision and attention dimensions
- Number of sequential steps
- Feature reuse coefficient
- Sparsity regularisation
- Learning rate and batch size

Model selection prioritises **F1-score and Recall**, ensuring sensitivity to the minority class.

---

### Stage II – Representation Pretraining and Robust Finetuning

This stage strengthens feature representations and inference robustness via:

**1. Self-Supervised Feature Reconstruction**
- Masked Feature Modelling (35% masking)
- Reconstruction loss combining MSE and cosine similarity

**2. Unsupervised TabNet Pretraining**
- Utilising TabNet Pretrainer to learn intrinsic feature dependencies

**3. Supervised Finetuning**
- AdamW optimiser with warm-up scheduling and early stopping
- Composite Focal–Dice loss for imbalance tolerance

**4. Robust Inference**
- Test-Time Augmentation (TTA)
- Dynamic decision threshold optimisation (τ\* = 0.34)

---

##  Experimental Setup

- **Dataset:** UCI Adult Census Income  
- **Task:** Binary classification (income > / ≤ $50K)  
- **Metrics:**  
  Accuracy, Precision, Recall, F1-score, ROC-AUC, MCC, Cohen’s Kappa, Average Precision

Baseline comparison is performed against a standard TabNet classifier trained with default settings.

---

##  Results

HT2P achieves consistent and statistically significant improvements over TabNet:

- **F1-score:** 0.6800 → 0.7046 (+3.62%)
- **Recall:** 0.6289 → 0.7649 (+21.63%)
- Improved probability calibration
- Reduced false negatives
- More balanced feature utilisation

### Overall Performance Comparison
<img width="5979" height="5485" alt="ultimate_tabnet_comparison" src="https://github.com/user-attachments/assets/453e6d76-6e3e-492e-b54a-03a0d5b74592" />


---

##  Feature Importance Analysis

HT2P learns fundamentally different and more distributed feature representations compared to TabNet, indicating deeper exploitation of feature interactions.
<img width="4765" height="3543" alt="feature_importance_analysis" src="https://github.com/user-attachments/assets/77b09322-e1c2-4ec8-a4f9-555f2eaa4ba2" />



---

##  Error Analysis and Model Behaviour

HT2P demonstrates:
- Lower prediction variance
- Better probabilistic calibration
- Reduced overconfidence on minority samples
<img width="4771" height="3541" alt="error_analysis_comparison" src="https://github.com/user-attachments/assets/0503ad40-c1ba-48bc-a03c-b1bcc52fcefd" />


---
