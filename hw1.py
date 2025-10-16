# -*- coding: utf-8 -*-
"""
@author: htchen

"""
import numpy as np
import numpy.linalg as la

def gram_schmidt(S1: np.ndarray):
    """
    Parameters
    ----------
    S1 : np.ndarray
        A m x n matrix with columns that need to be orthogonalized using Gram-Schmidt process.
        It is assumed that vectors in S = [v1 v2 ... vn] are linear independent.

    Returns
    -------
    S2 : np.ndarray
        S2 = [e1 e2 ... en] is a m x n orthogonal matrix such that span(S1) = span(S2)

    """
    m, n = S1.shape
    S2 = np.zeros((m, n))
    # write your code here

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

"""
Expected output:
------------------
S1 => 
[[ 7.  4.  7. -3. -9.]
 [-1. -4. -4.  1. -4.]
 [ 8.  0.  5. -6.  0.]
 [-4.  1.  1. -1.  4.]
 [ 2.  3. -5.  1.  8.]]
S2.T @ S2 => 
[[ 1. -0. -0.  0.  0.]
 [-0.  1. -0. -0. -0.]
 [-0. -0.  1.  0.  0.]
 [ 0. -0.  0.  1.  0.]
 [ 0. -0.  0.  0.  1.]]
"""  