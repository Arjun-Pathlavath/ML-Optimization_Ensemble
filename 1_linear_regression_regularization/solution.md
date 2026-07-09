# Machine Learning Assignment 1 – Detailed Solutions

Files used:
- Dataset 1: `salary_data - salary_data.csv`
- Dataset 2: `stats_females - stats_females.csv`
- Code: [solve_ass1.py](F:\SEM IV\lessons\ML\assaignments\Ass_1\solve_ass1.py)
- Figures: `figures/`
- Raw computed values: [ass1_results.json](F:\SEM IV\lessons\ML\assaignments\Ass_1\ass1_results.json)

---

## Q1. Linear regression using the closed-form solution

We use the normal equation with an intercept term:

\[
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
\]

For numerical stability, the implementation uses the Moore–Penrose pseudoinverse when needed.

### Q1.1 On the datasets provided

#### Dataset A: `salary_data - salary_data.csv`
- Model: \( y = w_0 + w_1 x \)
- Closed-form weights:
  - \( w_0 = 25792.2002 \)
  - \( w_1 = 9449.9623 \)
- scikit-learn (`LinearRegression`) weights match to numerical precision.

#### Dataset B: `stats_females - stats_females.csv`
- Target: `Height`
- Features: `momheight`, `dadheight`
- Model: \( y = w_0 + w_1 \cdot \text{momheight} + w_2 \cdot \text{dadheight} \)
- Closed-form weights:
  - \( w_0 = 18.5473 \)
  - \( w_1 = 0.3035 \)
  - \( w_2 = 0.3879 \)
- scikit-learn (`LinearRegression`) weights match to numerical precision.

### Q1.2 Given matrix \( X \) and vector \( y \)

\[
X = \begin{bmatrix}
1 & 2 \\
2 & 4 \\
3 & 6 \\
4 & 8
\end{bmatrix}, \quad
y = \begin{bmatrix}2\\3\\4\\5\end{bmatrix}
\]

After adding the intercept, \(X^T X\) becomes:

\[
X^T X =
\begin{bmatrix}
4 & 10 & 20 \\
10 & 30 & 60 \\
20 & 60 & 120
\end{bmatrix}
\]

- \( \det(X^T X) = 0 \)
- Rank \( = 2 \)

**Reason:** the two feature columns are linearly dependent (second column is exactly twice the first), so \(X^T X\) is singular. 

**Why scikit-learn still solves it:** scikit-learn uses SVD-based solvers (pseudoinverse), which can handle rank-deficient matrices and produce the minimum-norm least squares solution.

The pseudoinverse solution is:
- \( w_0 = 1.0 \)
- \( w_1 = 0.2 \)
- \( w_2 = 0.4 \)

---

## Q2. Batch, mini-batch, and stochastic gradient descent

We optimize:
\[
J(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N \left(y^{(i)} - (w_0 + w_1 x^{(i)})\right)^2
\]

Gradients:
\[
\frac{\partial J}{\partial w_0} = -\frac{2}{N}\sum_{i=1}^N \left(y^{(i)} - \hat{y}^{(i)}\right),
\quad
\frac{\partial J}{\partial w_1} = -\frac{2}{N}\sum_{i=1}^N \left(y^{(i)} - \hat{y}^{(i)}\right)x^{(i)}
\]

### Learning rate choice
The step size uses the Lipschitz constant of the quadratic loss:
\[
L = \frac{2}{N} \lambda_{\max}(X^T X), \quad \eta = 0.9 / L
\]

### Final weights
- **Batch GD** (\(\eta = 0.0122266\)):
  - \(w_0 = 25792.1964\), \(w_1 = 9449.9629\)
- **Mini-batch GD** (batch size 8, \(\eta = 0.0122266\), decayed):
  - \(w_0 = 25180.4108\), \(w_1 = 9539.7133\)
- **Stochastic GD** (\(\eta = 0.0012227\), decayed):
  - \(w_0 = 25959.3069\), \(w_1 = 9558.3252\)

Batch GD converges essentially to the closed-form solution. Mini-batch and SGD converge close but retain small noise due to stochasticity.

### Required plots
- Data + fitted lines: [q2_fit_lines.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q2_fit_lines.png)
- Error surface + GD path: [q2_surface_path.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q2_surface_path.png)
- Contour + line animation (200 ms per frame): [q2_contour_fit.gif](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q2_contour_fit.gif)

---

## Q3. Polynomial regression, bias–variance, and regularization

### Q3.1 Data generation
True function:
\[
 f(x) = x + 2\sin(0.5x)
\]

Data:
- \(x \in [-10, 10]\), 50 points
- noise: \(\mathcal{N}(0, 0.5)\)

Plot: [q3_data.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_data.png)

### Q3.2 Model selection (degree 0–10)
We fit polynomial regression using the closed-form least squares solution for each degree.

- All fitted curves: [q3_poly_fits.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_poly_fits.png)
- Training vs test error: [q3_error_vs_degree.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_error_vs_degree.png)

**Observation:**
- Low degrees (0–2) underfit (high bias).
- Medium degrees (around 6–8) fit best (low test error).
- Very high degrees show increased variance.

### Q3.3 Generalization error
Test set: 200 noise-free points from the true function. The test MSE vs degree is plotted and shows minimum around degrees 6–8.

### Q3.4 Bias–variance analysis
We use \(L=100\) bootstrap datasets, each with 50 points, and a fixed test set of 200 points.

Definitions used:
\[
\text{bias}^2 = \frac{1}{N} \sum_{n=1}^N (\bar{y}(x_n)-y(x_n))^2,
\quad
\text{variance} = \frac{1}{N} \sum_{n=1}^N \frac{1}{L} \sum_{l=1}^L (y^{(l)}(x_n)-\bar{y}(x_n))^2
\]

Plot: [q3_bias_variance.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_bias_variance.png)

**Observation:**
- Bias decreases with degree.
- Variance increases sharply at high degrees.
- The optimal tradeoff occurs around degrees 6–8.

### Q3.5 Overfitting and regularization

#### Overfitting (degree 4 in this run)
- The degree that maximizes the test–train gap is 4.
- Adding more data (100 points) stabilizes the fit.

Plot: [q3_overfit_more_data.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_overfit_more_data.png)

#### Lasso (L1) regression
We solve:
\[
J(w) = \frac{1}{N}\sum_{i=1}^N \left(y_i - (w_0 + w^T \phi(x_i))\right)^2 + \lambda \lVert w \rVert_1
\]

- Error vs \(\lambda\): [q3_lasso_errors.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_lasso_errors.png)
- Sparsity vs \(\lambda\): [q3_lasso_sparsity.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_lasso_sparsity.png)
- L1 norm vs \(\lambda\): [q3_lasso_l1norm.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_lasso_l1norm.png)

Best \(\lambda\) by test error: **0.0** (unregularized performs best on this draw). Lasso drives most coefficients to zero for larger \(\lambda\).

#### Ridge (L2) regression
We solve:
\[
J(w) = \frac{1}{N}\sum_{i=1}^N \left(y_i - (w_0 + w^T \phi(x_i))\right)^2 + \lambda \lVert w \rVert_2^2
\]

- Error vs \(\lambda\): [q3_ridge_errors.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_ridge_errors.png)
- L2 norm vs \(\lambda\): [q3_ridge_l2norm.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_ridge_l2norm.png)

Best \(\lambda\) by test error: **0.5**.

#### Comparison of curves
Plot: [q3_regularization_curves.png](F:\SEM IV\lessons\ML\assaignments\Ass_1\figures\q3_regularization_curves.png)

**Summary:**
- Lasso yields sparse coefficients and can eliminate higher-degree terms.
- Ridge shrinks coefficients smoothly without setting them to zero.

---

## How to reproduce

Run:

```powershell
$env:MPLCONFIGDIR="$env:TEMP\mpl"
python "F:\SEM IV\lessons\ML\assaignments\Ass_1\solve_ass1.py"
```

This regenerates all figures and the JSON summary.
