# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import cv2
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


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis = 0, keepdims=1)
m2 = np.mean(X2, axis = 0, keepdims=1)

# write you code here
X1c = X1 - m1
X2c = X2 - m2
S1 = X1c.T @ X1c
S2 = X2c.T @ X2c
Sw = S1 + S2

# Fisher LDA direction: w = Sw^{-1}(m1 - m2)
w = la.pinv(Sw) @ (m1 - m2).T
w = w / la.norm(w)  # normalize for plotting

# threshold for decision boundary (equal prior)
t = 0.5 * float(w.T @ (m1 + m2).T)

plt.figure(dpi=288)

plt.plot(X1[:, 0], X1[:,1], 'r.')
plt.plot(X2[:, 0], X2[:,1], 'g.')

# write you code here
mid = 0.5 * (m1 + m2)  # 1x2
L = 4.0
p1 = mid.flatten() - L * w.flatten()
p2 = mid.flatten() + L * w.flatten()
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)

# draw decision boundary: w^T x = t
w0, w1 = float(w[0, 0]), float(w[1, 0])
xmin = min(X1[:, 0].min(), X2[:, 0].min()) - 1
xmax = max(X1[:, 0].max(), X2[:, 0].max()) + 1
xs = np.linspace(xmin, xmax, 200)

if abs(w1) > 1e-12:
    ys = (t - w0 * xs) / w1
    plt.plot(xs, ys, 'k--', linewidth=2)
else:
    xline = t / w0
    ymin = min(X1[:, 1].min(), X2[:, 1].min()) - 1
    ymax = max(X1[:, 1].max(), X2[:, 1].max()) + 1
    plt.plot([xline, xline], [ymin, ymax], 'k--', linewidth=2)

plt.axis('equal')  
plt.show()

