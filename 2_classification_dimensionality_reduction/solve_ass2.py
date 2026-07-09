import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Matplotlib setup
os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TEMP', '.')) / 'mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression

ROOT = Path(r'F:\SEM IV\lessons\ML\assaignments\Ass_2')
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# --------------------------
# Load dataset
# --------------------------

# The file is CSV content with .xls extension
ads_path = ROOT / 'Social_Network_Ads.xls'
df = pd.read_csv(ads_path)

# Encode Gender
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Features and target
feature_cols = ['Gender', 'Age', 'EstimatedSalary']
X = df[feature_cols].values
y = df['Purchased'].values

# Standardize for distance-based models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split 90/10
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42, stratify=y
)

results = {
    'q1': {},
    'q2': {},
    'q3': {'status': 'dataset_missing'},
    'q4': {'status': 'dataset_missing'},
}

# --------------------------
# Q1: kNN classification
# --------------------------

# 3D visualization
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], X_scaled[y==0, 2], c='blue', label='0')
ax.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], X_scaled[y==1, 2], c='red', label='1')
ax.set_xlabel('Gender (scaled)')
ax.set_ylabel('Age (scaled)')
ax.set_zlabel('EstimatedSalary (scaled)')
ax.set_title('Social Network Ads (3D)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'q1_3d_scatter.png', dpi=200)
plt.close()

# kNN from scratch

def knn_predict(Xtr, ytr, Xte, k=5, metric='euclidean'):
    preds = []
    if metric == 'mahalanobis':
        cov = np.cov(Xtr.T)
        cov_inv = np.linalg.pinv(cov)
    for x in Xte:
        if metric == 'euclidean':
            d = np.linalg.norm(Xtr - x, axis=1)
        elif metric == 'cosine':
            num = Xtr @ x
            den = np.linalg.norm(Xtr, axis=1) * np.linalg.norm(x)
            d = 1 - (num / (den + 1e-12))
        elif metric == 'mahalanobis':
            diff = Xtr - x
            d = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        else:
            raise ValueError('Unknown metric')
        idx = np.argsort(d)[:k]
        vals, counts = np.unique(ytr[idx], return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

# Evaluate multiple k
k_values = list(range(1, 26))
metrics_k = {}
for k in k_values:
    y_pred = knn_predict(X_train, y_train, X_test, k=k, metric='euclidean')
    metrics_k[k] = compute_metrics(y_test, y_pred)

# Library comparison (sklearn)
knn_lib = KNeighborsClassifier(n_neighbors=5)
knn_lib.fit(X_train, y_train)
lib_pred = knn_lib.predict(X_test)
lib_metrics = compute_metrics(y_test, lib_pred)

# 5-fold cross-validation for k
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
for k in k_values:
    accs = []
    for train_idx, val_idx in kf.split(X_scaled):
        Xtr, Xval = X_scaled[train_idx], X_scaled[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        yhat = knn_predict(Xtr, ytr, Xval, k=k, metric='euclidean')
        accs.append(accuracy_score(yval, yhat))
    cv_acc.append(np.mean(accs))

best_k = k_values[int(np.argmax(cv_acc))]

plt.figure(figsize=(6,4))
plt.plot(k_values, cv_acc, marker='o')
plt.xlabel('k')
plt.ylabel('Average CV accuracy')
plt.title('Q1: 5-Fold CV Accuracy vs k')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q1_cv_accuracy.png', dpi=200)
plt.close()

# Compare metrics for best k and different distances
metrics_best = {}
for metric in ['euclidean', 'cosine', 'mahalanobis']:
    y_pred = knn_predict(X_train, y_train, X_test, k=best_k, metric=metric)
    metrics_best[metric] = compute_metrics(y_test, y_pred)

# Extreme k values
extreme_results = {}
for k in [1, len(X_train)]:
    y_pred = knn_predict(X_train, y_train, X_test, k=k, metric='euclidean')
    extreme_results[k] = compute_metrics(y_test, y_pred)

# Effect of training size with fixed k
train_size_results = {}
for train_frac in [0.4, 0.6, 0.9]:
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=1-train_frac, random_state=42, stratify=y)
    y_pred = knn_predict(Xtr, ytr, Xte, k=best_k, metric='euclidean')
    train_size_results[train_frac] = compute_metrics(yte, y_pred)

# Decision boundaries for different k (using feature pairs)
feature_pairs = [(0,1), (0,2), (1,2)]
ks_for_boundaries = [1, best_k, 15]

for (i, j) in feature_pairs:
    X2 = X_scaled[:, [i, j]]
    Xtr, Xte, ytr, yte = train_test_split(X2, y, test_size=0.1, random_state=42, stratify=y)
    x_min, x_max = X2[:,0].min() - 0.5, X2[:,0].max() + 0.5
    y_min, y_max = X2[:,1].min() - 0.5, X2[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for k in ks_for_boundaries:
        y_pred_grid = knn_predict(Xtr, ytr, grid, k=k, metric='euclidean')
        Z = y_pred_grid.reshape(xx.shape)
        plt.figure(figsize=(5,4))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.scatter(Xtr[ytr==0,0], Xtr[ytr==0,1], c='blue', s=15, label='0')
        plt.scatter(Xtr[ytr==1,0], Xtr[ytr==1,1], c='red', s=15, label='1')
        plt.xlabel(feature_cols[i])
        plt.ylabel(feature_cols[j])
        plt.title(f'kNN decision boundary (k={k})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f'q1_boundary_{feature_cols[i]}_{feature_cols[j]}_k{k}.png', dpi=200)
        plt.close()

# kNN regression (first two features -> third feature)
X_reg = X_scaled[:, :2]
y_reg = X_scaled[:, 2]
Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.4, random_state=42)

mse_by_k = {}
for k in k_values:
    knn_r = KNeighborsRegressor(n_neighbors=k)
    knn_r.fit(Xtr, ytr)
    yhat = knn_r.predict(Xte)
    mse_by_k[k] = mean_squared_error(yte, yhat)

plt.figure(figsize=(6,4))
plt.plot(list(mse_by_k.keys()), list(mse_by_k.values()), marker='o')
plt.xlabel('k')
plt.ylabel('MSE')
plt.title('Q1: kNN Regression MSE vs k')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q1_knn_regression_mse.png', dpi=200)
plt.close()

results['q1'] = {
    'k_metrics': metrics_k,
    'lib_metrics_k5': lib_metrics,
    'cv_accuracy': cv_acc,
    'best_k': best_k,
    'distance_metrics_best_k': metrics_best,
    'extreme_k_results': extreme_results,
    'train_size_results': train_size_results,
    'knn_regression_mse': mse_by_k,
}

# --------------------------
# Q2: Logistic regression
# --------------------------

# Add intercept
X_tr = np.hstack([np.ones((len(X_train),1)), X_train])
X_te = np.hstack([np.ones((len(X_test),1)), X_test])

# Sigmoid

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# SGD for logistic regression

def logistic_sgd(X, y, lr=0.1, epochs=2000):
    w = np.zeros(X.shape[1])
    for epoch in range(epochs):
        idx = np.random.permutation(len(X))
        for i in idx:
            xi = X[i]
            yi = y[i]
            p = sigmoid(xi @ w)
            grad = (p - yi) * xi
            w -= lr * grad
    return w

# Newton / IRLS

def logistic_newton(X, y, max_iter=50):
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = sigmoid(X @ w)
        W = np.diag(p * (1 - p))
        grad = X.T @ (p - y)
        H = X.T @ W @ X
        # use pseudo-inverse for stability
        step = np.linalg.pinv(H) @ grad
        w -= step
        if np.linalg.norm(step) < 1e-6:
            break
    return w

w_sgd = logistic_sgd(X_tr, y_train, lr=0.1, epochs=1000)
w_newton = logistic_newton(X_tr, y_train, max_iter=30)

# Library
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predictions
pred_sgd = (sigmoid(X_te @ w_sgd) >= 0.5).astype(int)
pred_newton = (sigmoid(X_te @ w_newton) >= 0.5).astype(int)
pred_lib = log_reg.predict(X_test)

results['q2'] = {
    'sgd': compute_metrics(y_test, pred_sgd),
    'newton': compute_metrics(y_test, pred_newton),
    'sklearn': compute_metrics(y_test, pred_lib),
}

# Decision boundary plot in 3D (sklearn)
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[y_train==0,0], X_train[y_train==0,1], X_train[y_train==0,2], c='blue', label='Train 0')
ax.scatter(X_train[y_train==1,0], X_train[y_train==1,1], X_train[y_train==1,2], c='red', label='Train 1')
ax.scatter(X_test[y_test==0,0], X_test[y_test==0,1], X_test[y_test==0,2], c='cyan', label='Test 0', marker='^')
ax.scatter(X_test[y_test==1,0], X_test[y_test==1,1], X_test[y_test==1,2], c='orange', label='Test 1', marker='^')

# Decision boundary plane: w0 + w1*x1 + w2*x2 + w3*x3 = 0
w = np.hstack([log_reg.intercept_, log_reg.coef_.flatten()])
# Solve for x3 = -(w0 + w1*x1 + w2*x2)/w3
x1_vals = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 10)
x2_vals = np.linspace(X_train[:,1].min(), X_train[:,1].max(), 10)
xx1, xx2 = np.meshgrid(x1_vals, x2_vals)
if abs(w[3]) > 1e-6:
    xx3 = -(w[0] + w[1]*xx1 + w[2]*xx2) / w[3]
    ax.plot_surface(xx1, xx2, xx3, alpha=0.3, color='gray')

ax.set_xlabel('Gender (scaled)')
ax.set_ylabel('Age (scaled)')
ax.set_zlabel('EstimatedSalary (scaled)')
ax.set_title('Logistic Regression Decision Boundary')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'q2_logreg_boundary.png', dpi=200)
plt.close()

# Save results
with open(ROOT / 'ass2_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print('Ass_2 results written to ass2_results.json and figures/')
