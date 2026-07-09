# Mathematical Optimization, Custom Classifiers, & Ensemble Systems

This repository contains a collection of core machine learning algorithms implemented, optimized, and analyzed from scratch. 

The project is structured into three primary sub-projects covering advanced optimization theory, classification, dimensionality reduction, and ensemble learning.

---

## 📂 Project Structure & Core Components

### 1. Linear Regression, Regularization & Bias-Variance Analysis
Located in: `1_linear_regression_regularization/`
* **Mathematical Optimization:** Implementation of the closed-form normal equation (with SVD-based pseudoinverse for singular matrices) and gradient descent variants (Batch GD, Mini-batch GD, SGD).
* **Lipschitz Constant Learning Rate:** Optimization of gradient descent step size ($\eta = 0.9/L$) using the **Lipschitz constant of the quadratic loss** ($L = \frac{2}{N}\lambda_{\max}(X^T X)$) to mathematically guarantee convergence.
* **Bias-Variance Study:** Bias-variance decomposition evaluated via **bootstrap resampling ($L=100$)** over polynomial degrees (0–10).
* **L1 & L2 Regularization:** Comparison of Lasso (L1) and Ridge (L2) regression, highlighting regularization paths, coefficient shrinkage, and model sparsity.

### 2. Non-Parametric Models, Custom Logistic Optimization & PCA
Located in: `2_classification_dimensionality_reduction/`
* **Custom kNN Classifier:** Multi-metric k-Nearest Neighbors supporting Euclidean, Cosine, and Mahalanobis distances, evaluated across hyper-parameters ($k = 1\dots25$) and training sizes.
* **Custom Logistic Solvers:** Implementation of Binary Logistic Regression from scratch using:
  1. **Iteratively Reweighted Least Squares (IRLS / Newton-Raphson method)** using Hessian curvature updates.
  2. **Stochastic Gradient Descent (SGD)**.
* **Principal Component Analysis (PCA):** Eigenface extraction and reconstruction error evaluation on the ORL Face Database.

### 3. Decision Trees & Ensemble Learning
Located in: `3_decision_trees_ensemble_learning/`
* **Decision Trees:** Recursive entropy-based tree splitting (Information Gain) for classification and residual sum of squares (RSS) for regression.
* **Bagging vs. Random Forests:** Comparative study of ensemble methods, highlighting the bias-variance trade-off and tree decorrelation through random feature subsampling.

---

## 📊 Key Results & Metrics

### Linear Regression & Regularization (Section 1)
* Batch Gradient Descent converges precisely to the closed-form normal equation.
* Polynomial evaluation shows that a degree **6–8** polynomial balances the bias-variance trade-off best, achieving a minimal test MSE of **~0.05**.

### Custom Classifiers & Optimization (Section 2)
On the `Social_Network_Ads` dataset:
* **Custom IRLS (Newton's Method)** and **Custom SGD** solvers both achieved **92.5% accuracy**, outperforming the default scikit-learn solver (**85.0%**) in our test split by **2.5%**.
* Custom kNN achieved a peak accuracy of **92.5%** at $k=6$ (Euclidean and Cosine distances). Mahalanobis distance achieved **87.5%**.

### Ensemble Learning Performance (Section 3)

#### **Car Evaluation (Classification Task)**
* **Decision Tree (Tuned Depth = 14):** 97.80% CV Accuracy
* **Bagging (Tuned B = 50):** **98.09%** CV Accuracy
* **Random Forest ($m=4$, $B=50$):** 96.35% CV Accuracy
* *Insight:* For small feature spaces (6 categorical features), Random Forest's feature restriction can exclude the dominant predictors, causing it to slightly underperform plain Bagging.

#### **Boston Housing (Regression Task)**
* **Bagging Regression ($B=250$):** 13.64 Mean Squared Error (MSE)
* **Random Forest Regression ($m=7$, $B=250$):** **9.97** Mean Squared Error (MSE)
* *Insight:* For larger feature spaces (13 continuous features), Random Forest achieves a **27.0% reduction in error** over Bagging by decorrelating base trees.

---

## 🛠️ How to Setup and Run

### 1. Set Up Virtual Environment & Dependencies
Ensure you have Python 3.10+ installed.
```bash
git clone <your-repo-url>
cd ml_optimization_ensemble
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn xlrd jupyter
```

### 2. Run the Projects
You can run the python scripts directly or launch Jupyter Notebook to explore:
```bash
# To run Section 1 script:
python 1_linear_regression_regularization/solve_ass1.py

# To run Section 2 script:
python 2_classification_dimensionality_reduction/solve_ass2.py

# To open Jupyter Notebooks:
jupyter notebook
```
Explore the notebooks in the subdirectories for detailed step-by-step mathematical breakdowns and diagnostic plots.
