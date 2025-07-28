# Machine Learning Exam Topics Analysis

## 1. Analysis of Past Exam (SS 2024)

This section breaks down the topics and point distribution from the Summer 2024 exam to provide context on core concepts.

- **Total Points:** 120
- **Duration:** 120 min

### Topic Distribution:
- **Neural Networks:** 36 points
- **Ridge & Logistic Regression:** 27 points
- **Decision Trees:** 19 points
- **Bias-Variance Tradeoff:** 15 points
- **Naive Bayes:** 13 points
- **SVM:** 12 points
- **Ensemble Methods (Random Forest, AdaBoost):** 2 points
- **Information Theory (Cross-Entropy):** 1 point

---

## 2. Consolidated Topics for Upcoming Exam

This is the definitive list of topics for the upcoming examination, combining concepts from the previous exam with the new material.

### 2.1. ML Foundations & Process
- **CRISP-DM:** Business/Data Understanding, Preparation, Modeling, Evaluation, Deployment.
- **Data Preprocessing:** Handling missing values, outlier detection, feature scaling (normalization/standardization), and feature engineering.
- **Evaluation:** Confusion matrix, Accuracy, Precision, Recall, F1-score.
- **Bias-Variance Tradeoff:** Decomposing error, analyzing models.
- **Bayes Error:** Understanding the irreducible error and the optimal Bayes classifier.

### 2.2. Supervised Learning
- **Linear & Ridge Regression:**
  - Linear Regression: Simple and multiple regression, feature selection, R² score.
  - Ridge Regression: L2 regularization (λ parameter), feature maps (φ(x)), and the kernel trick `k(x,y)`.
- **Logistic Regression:** Binary classification and multi-class classification (softmax), linear decision boundaries.
- **Naive Bayes:** Conditional independence assumption, Laplace smoothing, posterior probability calculation.
- **Decision Trees & Random Forest:**
  - Decision Trees: Entropy, Gini Impurity, Information Gain for splitting.
  - Random Forest: Ensemble of trees to reduce variance.
- **Support Vector Machines (SVM):** Hard/soft margins, the `C` parameter, and kernel functions (e.g., linear, polynomial, RBF).

### 2.3. Unsupervised Learning
- **Principal Component Analysis (PCA):** Dimensionality reduction, covariance matrix, eigenvectors/eigenvalues, explained variance.
- **K-Means Clustering:** Lloyd's algorithm (assignment and update steps), inertia, and initialization.
- **Expectation-Maximization (EM):** Algorithm for latent variables, particularly for Gaussian Mixture Models (GMM).

### 2.4. Neural Networks
- **Feed-Forward Neural Networks:** Architecture (layers, nodes), activation functions (ReLU, etc.), forward pass, backpropagation for gradient calculation, and gradient descent for weight updates.
- **Convolutional Neural Networks (CNN):** Convolutional and pooling layers, basic architectures.
- **PyTorch:** Foundational concepts, tensor operations, and automatic differentiation.

---

## 3. Predicted Topic Weights for Exam

Based on the analysis, here is a prediction of how the topics might be weighted in the exam.

### High Priority (15-25 points each):
1.  **Neural Networks (incl. CNN/PyTorch):** Likely to be a major, multi-part task.
2.  **PCA:** As a fundamental new topic, expect a computational question.
3.  **Clustering (K-Means/EM):** Another significant new area, likely with a theoretical or calculation-based task.
4.  **Regression (Linear/Ridge/Logistic):** Remains a cornerstone of the course.

### Medium Priority (10-15 points each):
5.  **Decision Trees & Random Forest**
6.  **SVM**
7.  **Naive Bayes & Bayes Classification**

### Lower Priority (5-10 points each):
8.  **Bias-Variance Tradeoff**
9.  **CRISP-DM & Data Preprocessing**
10. **Evaluation Metrics & Bayes Error**

---

## 4. Key Formulas to Master

### PCA
- **Covariance Matrix:** `Cov(X) = (1/n)XᵀX`
- **Principal Components:** `PC = X·v` (where `v` are eigenvectors of the covariance matrix)

### K-means
- **Objective Function:** `min Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²`
- **Centroid Update:** `μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x`

### EM Algorithm
- **E-step:** Calculate expectation of the log-likelihood `Q(θ|θ⁽ᵗ⁾) = E[log L(θ|X,Z)|X,θ⁽ᵗ⁾]`
- **M-step:** Find parameters that maximize the expectation `θ⁽ᵗ⁺¹⁾ = argmax Q(θ|θ⁽ᵗ⁾)`

### Neural Networks
- **Forward Pass:** `h = σ(Wx + b)`
- **Backward Pass (Chain Rule):** `∂L/∂W = ∂L/∂h · ∂h/∂W`

### Regression
- **Ridge Solution:** `β̂ = (XᵀX + λI)⁻¹Xᵀy`
- **Logistic (Multi-class):** `pᵢ = exp(wᵢᵀx) / Σⱼexp(wⱼᵀx)`

### Decision Trees
- **Entropy:** `H(S) = -Σ pᵢ·log₂(pᵢ)`
- **Information Gain:** `IG(S,A) = H(S) - Σ (|Sᵥ|/|S|)·H(Sᵥ)`

### SVM
- **Primal Objective:** `min ½||w||² + CΣξᵢ`
- **Constraint:** `yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ`

### Evaluation
- **Precision:** `TP / (TP + FP)`
- **Recall:** `TP / (TP + FN)`
- **F1-Score:** `2 * (Precision * Recall) / (Precision + Recall)`
