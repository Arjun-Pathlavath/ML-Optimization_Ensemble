# Machine Learning Assignment 2 – Detailed Solutions

Files used:
- Dataset: `Social_Network_Ads.xls` (CSV content)
- Code: [solve_ass2.py](F:\SEM IV\lessons\ML\assaignments\Ass_2\solve_ass2.py)
- Figures: `figures/`
- Raw computed values: [ass2_results.json](F:\SEM IV\lessons\ML\assaignments\Ass_2\ass2_results.json)

---

## Q1. kNN classification and regression

### Data preprocessing
- Features: `Gender`, `Age`, `EstimatedSalary`
- Target: `Purchased`
- `Gender` encoded as Male=1, Female=0
- Features are standardized for distance-based methods.
- Train/test split: 90%/10% (stratified)

### 3D visualization
Plot: [q1_3d_scatter.png](F:\SEM IV\lessons\ML\assaignments\Ass_2\figures\q1_3d_scatter.png)

### kNN from scratch vs library
- For k=5, the from-scratch and scikit-learn results match:
  - Accuracy = 0.925
  - Precision = 0.9231
  - Recall = 0.8571
  - F1 = 0.8889

### Confusion matrix metrics across k
Computed for k = 1…25 (see [ass2_results.json](F:\SEM IV\lessons\ML\assaignments\Ass_2\ass2_results.json)).

### 5-fold cross-validation
Average accuracy vs k:
- Best k = **6**
- Plot: [q1_cv_accuracy.png](F:\SEM IV\lessons\ML\assaignments\Ass_2\figures\q1_cv_accuracy.png)

### Distance metrics (best k=6)
- Euclidean: accuracy 0.925
- Cosine: accuracy 0.925
- Mahalanobis: accuracy 0.875

**Best:** Euclidean/Cosine (tie)

### Effect of extreme k
- k=1: accuracy 0.875 (high variance)
- k=360: accuracy 0.65 (predicts majority class)

### Effect of training size (fixed k=6)
- 40% train: accuracy 0.8917
- 60% train: accuracy 0.8938
- 90% train: accuracy 0.925

### Decision boundaries (feature pairs)
Generated for (Gender, Age), (Gender, EstimatedSalary), (Age, EstimatedSalary) and k in {1, 6, 15}:
- Example files:
  - [q1_boundary_Gender_Age_k1.png](F:\SEM IV\lessons\ML\assaignments\Ass_2\figures\q1_boundary_Gender_Age_k1.png)
  - [q1_boundary_Age_EstimatedSalary_k6.png](F:\SEM IV\lessons\ML\assaignments\Ass_2\figures\q1_boundary_Age_EstimatedSalary_k6.png)

**Observation:** larger k yields smoother decision boundaries.

### kNN regression
- Inputs: first two features (Gender, Age)
- Output: third feature (EstimatedSalary)
- Train/test split: 60%/40%

MSE vs k plot:
- [q1_knn_regression_mse.png](F:\SEM IV\lessons\ML\assaignments\Ass_2\figures\q1_knn_regression_mse.png)

---

## Q2. Logistic regression

### Objective
For binary classification with sigmoid:
\[
\sigma(z) = \frac{1}{1+e^{-z}}\quad,\quad h_w(x)=\sigma(w^T x)
\]
We minimize negative log-likelihood (cross-entropy).

### Optimization methods
- **Stochastic Gradient Descent** (from scratch)
- **Newton’s Method / IRLS** (from scratch)
- **scikit-learn LogisticRegression**

### Performance (90/10 split)
- SGD: Accuracy 0.875, F1 0.8
- Newton: Accuracy 0.875, F1 0.8
- scikit-learn: Accuracy 0.85, F1 0.75

### Decision boundary visualization
3D plot with plane boundary:
- [q2_logreg_boundary.png](F:\SEM IV\lessons\ML\assaignments\Ass_2\figures\q2_logreg_boundary.png)

---

## Q3. Multiclass logistic regression on 20 Newsgroups

The dataset from Kaggle is not present locally, so the experiment is prepared but not executed.

**To complete Q3:**
1. Download the dataset from Kaggle and place it under `Ass_2/data/20newsgroups/` (or adjust the path in code).
2. Run the provided script and update results in this report.

### Method outline (for when the dataset is available)
- Vectorize text using TF–IDF: \\(X \\in \\mathbb{R}^{N \\times D}\\)\n+- For K classes, use softmax:\n+\\[\n+P(y=k|x) = \\frac{\\exp(w_k^T x)}{\\sum_{j=1}^K \\exp(w_j^T x)}\n+\\]\n+- Minimize multinomial cross-entropy:\n+\\[\n+\\mathcal{L}(W) = -\\sum_{i=1}^N \\sum_{k=1}^K \\mathbb{1}[y_i=k] \\log P(y_i=k|x_i)\n+\\]\n+- For the imbalanced dataset, remove a subset of documents from selected classes and compare confusion matrices and overall accuracy against the balanced case.

---

## Q4. PCA on ORL Face Database

The ORL face image dataset is not present locally, so the experiment is prepared but not executed.

**To complete Q4:**
1. Place the ORL dataset under `Ass_2/data/orl_faces/` (standard 40 subjects × 10 images).
2. Run the PCA pipeline and regenerate plots and reconstruction metrics.

### Method outline (for when the dataset is available)
- Stack vectorized images to form \\(X \\in \\mathbb{R}^{10304 \\times 400}\\)\n+- Center: \\(\\tilde{X} = X - \\mu\\)\n+- Covariance: \\(S = \\tilde{X}^T \\tilde{X} \\in \\mathbb{R}^{400 \\times 400}\\)\n+- Eigendecompose \\(S = U \\Lambda U^T\\)\n+- Eigenfaces are columns of \\(U\\), reshaped to 112×92\n+- Project: \\(V = U^T \\tilde{X}\\), reconstruct: \\(\\hat{X} = U V + \\mu\\)\n+- Reconstruction error: \\(\\|X - \\hat{X}\\|_F\\)\n+- Select k by variance explained and reconstruction error tradeoff

---

## How to reproduce

Run:

```powershell
$env:MPLCONFIGDIR="$env:TEMP\mpl"
python "F:\SEM IV\lessons\ML\assaignments\Ass_2\solve_ass2.py"
```

This regenerates all figures and the JSON summary for Q1–Q2.
