# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:37:05 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    # if A is full rank, no lambda value is less than 1e-6 
    # append a small value to stop rank check
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

def row_norm_square(X):
    return np.sum(X * X, axis=1)

# gaussian weight array g=[ g_1 g_2 ... g_m ]
# g_i = exp(-0.5 * ||x_i - c||^2 / sigma^2)
def gaussian_weight(X, c, sigma=1.0):
    s = 0.5 / sigma / sigma;
    norm2 = row_norm_square(X - c)
    g = np.exp(-s * norm2)
    return g

# xt: a sample in Xt
# yt: predicted value of f(xt)
# yt = (X.T @ G(xt) @ X)^-1 @ X.T @ G(xt) @ y
def predict(X, y, Xt, sigma=1.0):
    ntest = Xt.shape[0] # number of test samples 
    yt = np.zeros(ntest)
    for xi in range(ntest):
        c = Xt[xi, :]
        g = gaussian_weight(X, c, sigma) # diagonal elements in G
        G = np.diag(g)
        w = la.pinv(X.T @ G @ X) @ X.T @ G @ y
        yt[xi] = c @ w
    return yt

# Xs: m x n matrix; 
# m: pieces of sample
# K: m x m kernel matrix
# K[i,j] = exp(-c(|xt_i|^2 + |xs_j|^2 -2(xt_i)^T @ xs_j)) where c = 0.5 / sigma^2
# 更多實作說明, 參考課程oneonte筆記

def calc_gaussian_kernel(Xt, Xs, sigma=1):
    nt, _ = Xt.shape # pieces of Xt
    ns, _ = Xs.shape # pieces of Xs
    
    norm_square = row_norm_square(Xt)
    F = np.tile(norm_square, (ns, 1)).T
    
    norm_square = row_norm_square(Xs)
    G = np.tile(norm_square, (nt, 1))
    
    E = F + G - 2.0 * Xt @ Xs.T
    s = 0.5 / (sigma * sigma)
    K = np.exp(-s * E)
    return K

# n: degree of polynomial
# generate X=[1 x x^2 x^3 ... x^n]
# m: pieces(rows) of data(X)
# X is a m x (n+1) matrix
def poly_data_matrix(x: np.ndarray, n: int):
    m = x.shape[0]
    X = np.zeros((m, n + 1))
    X[:, 0] = 1.0
    for deg in range(1, n + 1):
        X[:, deg] = X[:, deg - 1] * x
    return X

hw5_csv = pd.read_csv('data/hw5.csv')
hw5_dataset = hw5_csv.to_numpy(dtype = np.float64)

hours = hw5_dataset[:, 0]
sulfate = hw5_dataset[:, 1]


# 1) Plot raw data
plt.figure()
plt.plot(hours, sulfate, 'bo', markersize=4)
plt.title('concentration vs time')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True)

plt.figure()
plt.plot(hours, sulfate, 'bo', markersize=4)
plt.xscale("log")
plt.yscale("log")
plt.title('concentration vs time (log-log scale)')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True)

# 2) Prepare x/y (column vectors)
x = hours.reshape(-1, 1)     # m x 1
y = sulfate.reshape(-1, 1)   # m x 1
m = x.shape[0]

x_min, x_max = float(hours.min()), float(hours.max())
xt = np.linspace(x_min, x_max, 300).reshape(-1, 1)

# 3) Polynomial regression (least squares)
#    X = [1, x, x^2, ..., x^n]
deg = 3
X = poly_data_matrix(hours, deg)           
Xt_poly = poly_data_matrix(xt.flatten(), deg)

w_poly = la.pinv(X) @ y    
y_poly = (Xt_poly @ w_poly).flatten()

plt.figure()
plt.plot(hours, sulfate, 'bo', markersize=4, label='data')
plt.plot(xt.flatten(), y_poly, 'r-', linewidth=2, label=f'poly fit (deg={deg})')
plt.title('Polynomial Least Squares Fit')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True)
plt.legend()

print("===== Polynomial regression =====")
print(f"degree = {deg}")
for i, wi in enumerate(w_poly.flatten()):
    print(f"w[{i}] = {wi:.6e}")


# 4) Locally Weighted Linear Regression (your predict)
sigma_lw = 50.0 
X_lin = np.hstack([np.ones((m, 1)), x])       # m x 2  => [1, x]
Xt_lin = np.hstack([np.ones((xt.shape[0], 1)), xt])  # nt x 2

y_lw = predict(X_lin, y, Xt_lin, sigma=sigma_lw)

plt.figure()
plt.plot(hours, sulfate, 'bo', markersize=4, label='data')
plt.plot(xt.flatten(), y_lw, 'g-', linewidth=2, label=f'LWLR (sigma={sigma_lw})')
plt.title('Locally Weighted Linear Regression')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True)
plt.legend()

# 5) Gaussian Kernel Ridge Regression (KRR)
#    alpha = (K + lam I)^-1 y
#    y(xt) = k(xt, x)^T alpha
sigma_k = 50.0
lam = 1e-3

K = calc_gaussian_kernel(x, x, sigma=sigma_k)                 # m x m
alpha = la.solve(K + lam * np.eye(m), y)                      # m x 1

Kt = calc_gaussian_kernel(xt, x, sigma=sigma_k)               # nt x m
y_krr = (Kt @ alpha).flatten()

plt.figure()
plt.plot(hours, sulfate, 'bo', markersize=4, label='data')
plt.plot(xt.flatten(), y_krr, 'm-', linewidth=2, label=f'KRR (sigma={sigma_k}, lam={lam})')
plt.title('Gaussian Kernel Ridge Regression')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True)
plt.legend()

plt.show()



