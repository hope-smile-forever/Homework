# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:37:05 2021
@author: htchen
"""

# Safe reset section (works in Spyder and VS Code)
try:
    from IPython import get_ipython
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('reset', '-sf')
except:
    pass

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2

plt.rcParams['figure.dpi'] = 144 


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
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

img = cv2.imread('data/svd_demo1.jpg', cv2.IMREAD_GRAYSCALE)
A = img.astype(dtype=np.float64)

U, Sigma, V = mysvd(A)
VT = V.T

def compute_energy(X: np.ndarray):
    return np.sum(X ** 2)
    
img_h, img_w = A.shape
keep_r = 201
rs = np.arange(1, keep_r)

energy_A = compute_energy(A)
energy_N = np.zeros(keep_r)

for r in rs:
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :]
    Noise = A - A_bar 
    energy_N[r] = compute_energy(Noise)

SNR = 10 * np.log10(energy_A / energy_N[1:])

plt.figure()
plt.plot(rs[1:], SNR)
plt.title('SNR vs. r')
plt.xlabel('r')
plt.ylabel('SNR (dB)')
plt.grid(True)
plt.show()

lambdas, _ = myeig(A.T @ A, symmetric=True)
lambdas = np.real(lambdas)

for r in [5, 20, 50, 100]:
    noise_energy_sum = np.sum(lambdas[r:])
    print(f"r={r:3d} => energy_N[r]={energy_N[r]:.4e}, sum(lambda[r+1:])={noise_energy_sum:.4e}")
