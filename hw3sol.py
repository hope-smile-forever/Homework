# -*- coding: utf-8 -*-
"""
Fourier basis least-squares fitting (square wave) using SVD
step1: build design matrix X
step2: SVD: X = U Sigma V^T
step3: least-squares solution a = V Sigma^{-1} U^T y
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Provided: eigen / SVD
# -----------------------------
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1]
    return lambdas[sorted_idx], V[:, sorted_idx]

def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)

    # numerical rank detection
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()

    lambdas, V = lambdas[:rank], V[:, :rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

# -----------------------------
# Data: square wave
# -----------------------------
pts = 50
x = np.linspace(-2, 2, pts)
y = np.zeros_like(x)

pts2 = pts // 2
y[:pts2] = -1
y[pts2:] = 1

# sort by x (optional, but keep as original)
idx = np.argsort(x)
x = x[idx]
y = y[idx]

T0 = np.max(x) - np.min(x)
f0 = 1.0 / T0
omega0 = 2.0 * np.pi * f0

# =============================
# step1: generate design matrix X
# X = [1, cos(w0*x), cos(2w0*x), ... cos(nw0*x), sin(w0*x), ... sin(nw0*x)]
# =============================
n = 10   # number of harmonics (try 5, 10, 20...)

X = np.zeros((pts, 1 + 2*n), dtype=float)
X[:, 0] = 1.0  # bias term

for k in range(1, n+1):
    X[:, k] = np.cos(k * omega0 * x)        # cos terms
    X[:, n + k] = np.sin(k * omega0 * x)    # sin terms

# =============================
# step2: SVD of X  ->  X = U Sigma V^T
# =============================
U, Sigma, V = mysvd(X)     # Sigma is r x r

# =============================
# step3: least squares solution
# a = X^+ y = V Sigma^{-1} U^T y
# =============================
s = np.diag(Sigma)                         # singular values (length r)
s_inv = 1.0 / s                            # since rank check removed tiny ones
Sigma_inv = np.diag(s_inv)

a = V @ Sigma_inv @ (U.T @ y)              # coefficient vector

# prediction
y_bar = X @ a

# -----------------------------
# Print coefficients (like homework output)
# -----------------------------
bias = a[0]
a_cos = a[1:n+1]
a_sin = a[n+1:2*n+1]

print("===== Fourier LS coefficients =====")
print(f"n (harmonics) = {n}")
print(f"bias a0 = {bias:.6f}")

print("\ncos coefficients:")
for k in range(1, n+1):
    print(f"a_cos[{k}] = {a_cos[k-1]: .6f}")

print("\nsin coefficients:")
for k in range(1, n+1):
    print(f"a_sin[{k}] = {a_sin[k-1]: .6f}")

# -----------------------------
# Plot
# -----------------------------
plt.plot(x, y_bar, 'g-', label='predicted values')
plt.plot(x, y, 'b-', label='true values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
