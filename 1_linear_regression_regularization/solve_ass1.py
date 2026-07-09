import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

# Matplotlib setup (avoid permission issues)
os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TEMP', '.')) / 'mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(r'F:\SEM IV\lessons\ML\assaignments\Ass_1')
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# --------------------------
# Q1: Closed-form regression
# --------------------------

def normal_eq(X, y):
    XTX = X.T @ X
    XTX_pinv = np.linalg.pinv(XTX)
    w = XTX_pinv @ X.T @ y
    return w, XTX

results = {
    'q1': {},
    'q2': {},
    'q3': {}
}

# Dataset 1: salary
salary_path = ROOT / 'salary_data - salary_data.csv'
salary_df = pd.read_csv(salary_path)
X_salary = salary_df[['YearsExperience']].values
y_salary = salary_df['Salary'].values.reshape(-1, 1)
X_salary_i = np.hstack([np.ones((len(X_salary), 1)), X_salary])

w_salary, XTX_salary = normal_eq(X_salary_i, y_salary)

lr_salary = LinearRegression(fit_intercept=True)
lr_salary.fit(X_salary, y_salary)

results['q1']['salary'] = {
    'w_normal_eq': w_salary.flatten().tolist(),
    'w_sklearn': [lr_salary.intercept_.item(), lr_salary.coef_.flatten()[0].item()],
}

# Dataset 2: stats_females
stats_path = ROOT / 'stats_females - stats_females.csv'
stats_df = pd.read_csv(stats_path)
X_stats = stats_df[['momheight', 'dadheight']].values
# target = Height
y_stats = stats_df['Height'].values.reshape(-1, 1)
X_stats_i = np.hstack([np.ones((len(X_stats), 1)), X_stats])

w_stats, XTX_stats = normal_eq(X_stats_i, y_stats)

lr_stats = LinearRegression(fit_intercept=True)
lr_stats.fit(X_stats, y_stats)

results['q1']['stats_females'] = {
    'w_normal_eq': w_stats.flatten().tolist(),
    'w_sklearn': [lr_stats.intercept_.item()] + lr_stats.coef_.flatten().tolist(),
}

# Q1 part 2: given X and y
X_given = np.array([[1,2],[2,4],[3,6],[4,8]], dtype=float)
y_given = np.array([2,3,4,5], dtype=float).reshape(-1,1)
X_given_i = np.hstack([np.ones((len(X_given), 1)), X_given])

w_given, XTX_given = normal_eq(X_given_i, y_given)

lr_given = LinearRegression(fit_intercept=True)
lr_given.fit(X_given, y_given)

results['q1']['given'] = {
    'XTX': XTX_given.tolist(),
    'det_XTX': float(np.linalg.det(XTX_given)),
    'rank_XTX': int(np.linalg.matrix_rank(XTX_given)),
    'w_normal_eq_pinv': w_given.flatten().tolist(),
    'w_sklearn': [lr_given.intercept_.item()] + lr_given.coef_.flatten().tolist(),
}

# --------------------------
# Q2: Gradient Descent
# --------------------------

x = X_salary.flatten()
y = y_salary.flatten()
N = len(x)

def mse(w0, w1, x, y):
    yhat = w0 + w1 * x
    return np.mean((y - yhat) ** 2)

def gd_batch(x, y, lr, max_iter=5000, tol=1e-8):
    w0 = 0.0
    w1 = 0.0
    history = []
    prev = None
    for it in range(max_iter):
        yhat = w0 + w1 * x
        err = y - yhat
        grad0 = (-2.0 / len(x)) * np.sum(err)
        grad1 = (-2.0 / len(x)) * np.sum(err * x)
        w0 -= lr * grad0
        w1 -= lr * grad1
        j = np.mean(err ** 2)
        history.append((w0, w1, j))
        if prev is not None and abs(prev - j) < tol:
            break
        prev = j
    return w0, w1, history


def gd_minibatch(x, y, lr, batch_size=8, max_iter=8000, tol=1e-8):
    w0 = 0.0
    w1 = 0.0
    history = []
    prev = None
    n = len(x)
    for it in range(max_iter):
        lr_t = lr / math.sqrt(it + 1)
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start:start+batch_size]
            xb = x[batch]
            yb = y[batch]
            yhat = w0 + w1 * xb
            err = yb - yhat
            grad0 = (-2.0 / len(xb)) * np.sum(err)
            grad1 = (-2.0 / len(xb)) * np.sum(err * xb)
            w0 -= lr_t * grad0
            w1 -= lr_t * grad1
        j = mse(w0, w1, x, y)
        history.append((w0, w1, j))
        if prev is not None and abs(prev - j) < tol:
            break
        prev = j
    return w0, w1, history


def gd_stochastic(x, y, lr, max_iter=200000):
    w0 = 0.0
    w1 = 0.0
    history = []
    n = len(x)
    for it in range(max_iter):
        i = np.random.randint(0, n)
        xi = x[i]
        yi = y[i]
        yhat = w0 + w1 * xi
        err = yi - yhat
        grad0 = -2.0 * err
        grad1 = -2.0 * err * xi
        lr_t = lr / (1.0 + 1e-5 * it)
        w0 -= lr_t * grad0
        w1 -= lr_t * grad1
        if it % n == 0:
            j = mse(w0, w1, x, y)
            history.append((w0, w1, j))
    return w0, w1, history


# Learning rates based on Lipschitz constant of the quadratic loss
X_i = np.hstack([np.ones((len(x), 1)), x.reshape(-1, 1)])
eigvals = np.linalg.eigvals(X_i.T @ X_i)
L = (2.0 / len(x)) * np.max(np.real(eigvals))
lr_base = 0.9 / L

batch_lr = float(lr_base)
mini_lr = float(lr_base)  # same scale works well for this dataset
sgd_lr = float(lr_base * 0.1)

w0_b, w1_b, hist_b = gd_batch(x, y, batch_lr, max_iter=5000)
w0_m, w1_m, hist_m = gd_minibatch(x, y, mini_lr, batch_size=8, max_iter=8000)
w0_s, w1_s, hist_s = gd_stochastic(x, y, sgd_lr, max_iter=200000)

results['q2'] = {
    'batch_lr': batch_lr,
    'mini_lr': mini_lr,
    'sgd_lr': sgd_lr,
    'batch_w': [w0_b, w1_b],
    'mini_w': [w0_m, w1_m],
    'sgd_w': [w0_s, w1_s],
    'batch_iters': len(hist_b),
    'mini_iters': len(hist_m),
    'sgd_iters': len(hist_s),
}

# Plot data and hypothesis lines
x_line = np.linspace(x.min(), x.max(), 200)
plt.figure(figsize=(6,4))
plt.scatter(x, y, color='black', s=20, label='Data')
plt.plot(x_line, w0_b + w1_b * x_line, label='Batch GD')
plt.plot(x_line, w0_m + w1_m * x_line, label='Mini-batch GD')
plt.plot(x_line, w0_s + w1_s * x_line, label='SGD')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Q2: Linear Fit via Gradient Descent')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'q2_fit_lines.png', dpi=200)
plt.close()

# 3D mesh plot with GD path (batch)
path = np.array(hist_b)
if len(path) > 200:
    path = path[::max(1, len(path)//200)]

w0_vals = np.linspace(w0_b - 20000, w0_b + 20000, 60)
w1_vals = np.linspace(w1_b - 3000, w1_b + 3000, 60)
W0, W1 = np.meshgrid(w0_vals, w1_vals)
J = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        J[i,j] = mse(W0[i,j], W1[i,j], x, y)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, J, cmap=cm.viridis, alpha=0.75)
ax.plot(path[:,0], path[:,1], path[:,2], color='red', marker='o', markersize=2, linewidth=1)
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('J(w)')
ax.set_title('Q2: Error Surface with Batch GD Path')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q2_surface_path.png', dpi=200)
plt.close()

# Animation: contour + fit line
# Use a shorter path for speed
path_anim = np.array(hist_b)
if len(path_anim) > 200:
    path_anim = path_anim[::max(1, len(path_anim)//200)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

# Contour
cs = ax1.contour(W0, W1, J, levels=30, cmap=cm.viridis)
ax1.clabel(cs, inline=1, fontsize=7)
ax1.set_xlabel('w0')
ax1.set_ylabel('w1')

# Data scatter
ax2.scatter(x, y, color='black', s=20)
ax2.set_xlabel('YearsExperience')
ax2.set_ylabel('Salary')

point, = ax1.plot([], [], 'ro')
arrow_line, = ax1.plot([], [], color='red')
line, = ax2.plot([], [], color='red')

def init():
    point.set_data([], [])
    arrow_line.set_data([], [])
    line.set_data([], [])
    return point, arrow_line, line

x_line = np.linspace(x.min(), x.max(), 200)


def update(frame):
    w0, w1, j = path_anim[frame]
    point.set_data([w0], [w1])
    if frame < len(path_anim)-1:
        w0n, w1n, _ = path_anim[frame+1]
    else:
        w0n, w1n = w0, w1
    arrow_line.set_data([w0, w0n], [w1, w1n])
    line.set_data(x_line, w0 + w1 * x_line)
    fig.suptitle(f'Iteration {frame+1}, RSS={j:.2f}')
    return point, arrow_line, line

anim = FuncAnimation(fig, update, frames=len(path_anim), init_func=init, blit=False, interval=200)
anim.save(FIG_DIR / 'q2_contour_fit.gif', writer=PillowWriter(fps=5))
plt.close(fig)

# --------------------------
# Q3: Polynomial regression
# --------------------------

def true_f(x):
    return x + 2*np.sin(0.5*x)

# Data generation
x_train = np.linspace(-10, 10, 50)
noise = np.random.normal(0, math.sqrt(0.5), size=x_train.shape)
y_train = true_f(x_train) + noise

x_test = np.linspace(-10, 10, 200)
y_test = true_f(x_test)

# Plot data and true function
plt.figure(figsize=(6,4))
plt.scatter(x_train, y_train, s=20, color='black', label='Noisy data')
plt.plot(x_test, y_test, color='blue', label='True function')
plt.legend()
plt.title('Q3: Generated Data')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_data.png', dpi=200)
plt.close()

train_errors = []
test_errors = []

# Plot learned curves for degrees 0..10
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
axes = axes.flatten()

for degree in range(0, 11):
    poly = PolynomialFeatures(degree, include_bias=True)
    Xtr = poly.fit_transform(x_train.reshape(-1,1))
    Xte = poly.transform(x_test.reshape(-1,1))
    w = np.linalg.pinv(Xtr) @ y_train
    yhat_tr = Xtr @ w
    yhat_te = Xte @ w
    tr_mse = np.mean((y_train - yhat_tr)**2)
    te_mse = np.mean((y_test - yhat_te)**2)
    train_errors.append(tr_mse)
    test_errors.append(te_mse)

    ax = axes[degree]
    ax.scatter(x_train, y_train, s=10, color='black')
    ax.plot(x_test, y_test, color='blue')
    ax.plot(x_test, yhat_te, color='red')
    ax.set_title(f'degree {degree}')

# hide any unused subplot
for k in range(11, len(axes)):
    axes[k].axis('off')

plt.suptitle('Q3: Polynomial Fits (degree 0-10)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_poly_fits.png', dpi=200)
plt.close()

# Error vs degree
plt.figure(figsize=(6,4))
plt.plot(range(0, 11), train_errors, marker='o', label='Train MSE')
plt.plot(range(0, 11), test_errors, marker='o', label='Test MSE')
plt.xlabel('Polynomial degree')
plt.ylabel('MSE')
plt.title('Q3: Training vs Test Error')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_error_vs_degree.png', dpi=200)
plt.close()

# Bias-variance using bootstrap
L = 100
Ntest = len(x_test)

bias2 = []
variance = []
train_err_boot = []
test_err_boot = []

for degree in range(0, 11):
    poly = PolynomialFeatures(degree, include_bias=True)
    preds = []
    tr_errs = []
    te_errs = []
    for _ in range(L):
        x_b = np.random.uniform(-10, 10, 50)
        y_b = true_f(x_b) + np.random.normal(0, math.sqrt(0.5), size=x_b.shape)
        Xb = poly.fit_transform(x_b.reshape(-1,1))
        Xt = poly.transform(x_test.reshape(-1,1))
        w = np.linalg.pinv(Xb) @ y_b
        yhat_tr = Xb @ w
        yhat_te = Xt @ w
        preds.append(yhat_te)
        tr_errs.append(np.mean((y_b - yhat_tr)**2))
        te_errs.append(np.mean((y_test - yhat_te)**2))
    preds = np.array(preds)  # L x Ntest
    ybar = preds.mean(axis=0)
    bias2.append(np.mean((ybar - y_test)**2))
    variance.append(np.mean(np.var(preds, axis=0)))
    train_err_boot.append(np.mean(tr_errs))
    test_err_boot.append(np.mean(te_errs))

plt.figure(figsize=(6,4))
plt.plot(range(0, 11), bias2, marker='o', label='Bias^2')
plt.plot(range(0, 11), variance, marker='o', label='Variance')
plt.plot(range(0, 11), train_err_boot, marker='o', label='Train MSE')
plt.plot(range(0, 11), test_err_boot, marker='o', label='Test MSE')
plt.xlabel('Polynomial degree')
plt.ylabel('Value')
plt.title('Q3: Bias-Variance Tradeoff')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_bias_variance.png', dpi=200)
plt.close()

# Identify overfitting degree: max (test - train)
overfit_degree = int(np.argmax(np.array(test_errors) - np.array(train_errors)))

# Using more data
poly_over = PolynomialFeatures(overfit_degree, include_bias=True)
Xtr_over = poly_over.fit_transform(x_train.reshape(-1,1))
w_over = np.linalg.pinv(Xtr_over) @ y_train
Xte_over = poly_over.transform(x_test.reshape(-1,1))
yhat_over = Xte_over @ w_over

x_train_more = np.linspace(-10, 10, 100)
noise_more = np.random.normal(0, math.sqrt(0.5), size=x_train_more.shape)
y_train_more = true_f(x_train_more) + noise_more
Xtr_more = poly_over.fit_transform(x_train_more.reshape(-1,1))
w_more = np.linalg.pinv(Xtr_more) @ y_train_more
yhat_more = Xte_over @ w_more

plt.figure(figsize=(6,4))
plt.plot(x_test, y_test, color='blue', label='True function')
plt.plot(x_test, yhat_over, color='red', label=f'Fit deg {overfit_degree} (50 pts)')
plt.plot(x_test, yhat_more, color='green', label=f'Fit deg {overfit_degree} (100 pts)')
plt.scatter(x_train, y_train, s=10, color='black', alpha=0.5)
plt.legend()
plt.title('Q3: Overfitting and More Data')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_overfit_more_data.png', dpi=200)
plt.close()

# Lasso and Ridge
lambdas = np.arange(0, 5.0 + 0.5, 0.5)

lasso_train = []
lasso_test = []
lasso_nz = []
lasso_l1 = []

ridge_train = []
ridge_test = []
ridge_l2 = []

poly = PolynomialFeatures(overfit_degree, include_bias=True)
Xtr = poly.fit_transform(x_train.reshape(-1,1))
Xte = poly.transform(x_test.reshape(-1,1))

for lam in lambdas:
    # Lasso
    lasso = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=lam, max_iter=20000))
    lasso.fit(Xtr, y_train)
    yhat_tr = lasso.predict(Xtr)
    yhat_te = lasso.predict(Xte)
    lasso_train.append(np.mean((y_train - yhat_tr)**2))
    lasso_test.append(np.mean((y_test - yhat_te)**2))
    coef = lasso.named_steps['lasso'].coef_
    lasso_nz.append(int(np.sum(np.abs(coef) > 1e-6)))
    lasso_l1.append(float(np.sum(np.abs(coef))))

    # Ridge
    ridge = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=lam))
    ridge.fit(Xtr, y_train)
    yhat_tr = ridge.predict(Xtr)
    yhat_te = ridge.predict(Xte)
    ridge_train.append(np.mean((y_train - yhat_tr)**2))
    ridge_test.append(np.mean((y_test - yhat_te)**2))
    coef_r = ridge.named_steps['ridge'].coef_
    ridge_l2.append(float(np.sqrt(np.sum(coef_r**2))))

best_lasso = int(np.argmin(lasso_test))
best_ridge = int(np.argmin(ridge_test))

# Lasso plots
plt.figure(figsize=(6,4))
plt.plot(lambdas, lasso_train, marker='o', label='Train MSE')
plt.plot(lambdas, lasso_test, marker='o', label='Test MSE')
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.title('Q3: Lasso Errors')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_lasso_errors.png', dpi=200)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(lambdas, lasso_nz, marker='o')
plt.xlabel('lambda')
plt.ylabel('Non-zero coefficients')
plt.title('Q3: Lasso Sparsity')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_lasso_sparsity.png', dpi=200)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(lambdas, lasso_l1, marker='o')
plt.xlabel('lambda')
plt.ylabel('L1 norm')
plt.title('Q3: Lasso L1 Norm')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_lasso_l1norm.png', dpi=200)
plt.close()

# Ridge plots
plt.figure(figsize=(6,4))
plt.plot(lambdas, ridge_train, marker='o', label='Train MSE')
plt.plot(lambdas, ridge_test, marker='o', label='Test MSE')
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.title('Q3: Ridge Errors')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_ridge_errors.png', dpi=200)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(lambdas, ridge_l2, marker='o')
plt.xlabel('lambda')
plt.ylabel('L2 norm')
plt.title('Q3: Ridge L2 Norm')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_ridge_l2norm.png', dpi=200)
plt.close()

# Compare curves for best lambda
lasso_best = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=lambdas[best_lasso], max_iter=20000))
lasso_best.fit(Xtr, y_train)
lasso_curve = lasso_best.predict(Xte)

ridge_best = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=lambdas[best_ridge]))
ridge_best.fit(Xtr, y_train)
ridge_curve = ridge_best.predict(Xte)

# Unregularized curve
w_unreg = np.linalg.pinv(Xtr) @ y_train
unreg_curve = Xte @ w_unreg

plt.figure(figsize=(6,4))
plt.plot(x_test, y_test, color='blue', label='True function')
plt.plot(x_test, unreg_curve, color='red', label='Unregularized')
plt.plot(x_test, lasso_curve, color='green', label=f'Lasso (λ={lambdas[best_lasso]:.1f})')
plt.plot(x_test, ridge_curve, color='purple', label=f'Ridge (λ={lambdas[best_ridge]:.1f})')
plt.scatter(x_train, y_train, s=10, color='black', alpha=0.4)
plt.legend()
plt.title('Q3: Regularization Comparison')
plt.tight_layout()
plt.savefig(FIG_DIR / 'q3_regularization_curves.png', dpi=200)
plt.close()

results['q3'] = {
    'train_errors': train_errors,
    'test_errors': test_errors,
    'bias2': bias2,
    'variance': variance,
    'train_err_boot': train_err_boot,
    'test_err_boot': test_err_boot,
    'overfit_degree': overfit_degree,
    'best_lasso_lambda': float(lambdas[best_lasso]),
    'best_ridge_lambda': float(lambdas[best_ridge]),
    'lasso_nz': lasso_nz,
}

# Save results
with open(ROOT / 'ass1_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print('Ass_1 results written to ass1_results.json and figures/')
