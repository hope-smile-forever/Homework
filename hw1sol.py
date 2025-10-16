# -*- coding: utf-8 -*-
"""
@author: htchen
"""
import numpy as np
import numpy.linalg as la

def gram_schmidt(S1: np.ndarray):
    m, n = S1.shape
    S2 = np.zeros((m, n))
    for i in range(n):
        vi = S1[:, i]
        ui = vi.copy()
        for j in range(i):
            ej = S2[:, j]
            ui -= np.dot(vi, ej) * ej
        norm_ui = la.norm(ui)
        if norm_ui > 1e-12:
            S2[:, i] = ui / norm_ui
    return S2

S1 = np.array([[ 7,  4,  7, -3, -9],
               [-1, -4, -4,  1, -4],
               [ 8,  0,  5, -6,  0],
               [-4,  1,  1, -1,  4],
               [ 2,  3, -5,  1,  8]], dtype=np.float64)
S2 = gram_schmidt(S1)

np.set_printoptions(precision=2, suppress=True)
print(f'S1 => \n{S1}')
print(f'S2.T @ S2 => \n{S2.T @ S2}')
