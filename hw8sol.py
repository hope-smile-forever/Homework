# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd


hw8_csv = pd.read_csv('data/hw8.csv')
hw8_dataset = hw8_csv.to_numpy(dtype = np.float64)

X0 = hw8_dataset[:, 0:2]
y = hw8_dataset[:, 2]

# write your code here
# add bias term: X = [1, x1, x2]
m = X0.shape[0]
X = np.hstack([np.ones((m, 1)), X0])   # m x 3

# linear classifier by least squares: w = argmin ||Xw - y||^2
w = la.pinv(X) @ y                     # (3,)

# make grid for decision region coloring
x1_min, x1_max = X0[:, 0].min() - 0.5, X0[:, 0].max() + 0.5
x2_min, x2_max = X0[:, 1].min() - 0.5, X0[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 400),
                       np.linspace(x2_min, x2_max, 400))

Xg = np.c_[np.ones(xx1.size), xx1.ravel(), xx2.ravel()]   # (N,3)
zg = (Xg @ w).reshape(xx1.shape)                          # decision value


fig = plt.figure(dpi=288)

# write your code here
# decision regions (coloring)
plt.contourf(xx1, xx2, zg, levels=[-1e9, 0, 1e9], alpha=0.25)

# decision boundary: w0 + w1*x1 + w2*x2 = 0
plt.contour(xx1, xx2, zg, levels=[0], colors='k', linewidths=2)

# 畫出分類邊界線及著色 

plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.show()

