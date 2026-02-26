import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def gaussian_K(x1,x2):
	return x1@x2


def optimize(X,y,data_length,idx_1,idx_2,alpha, C):
	zeta=0
	for i in range(data_length):
		if i != idx_1 and i != idx_2:
			zeta += (alpha[i]*y[i])
	zeta*=(-1)
	a=-1/2*(gaussian_K(X[idx_1],X[idx_1])+gaussian_K(X[idx_2],X[idx_2])-2*gaussian_K(X[idx_1],X[idx_2]))

	sig=0
	for i in range(data_length):
		if i != idx_1 and i != idx_2:
			sig = sig + (alpha[i]*y[i])*(gaussian_K(X[idx_1],X[i])-gaussian_K(X[idx_2],X[i]))
	b=1-y[idx_1]*y[idx_2]+y[idx_2]*sig-zeta*gaussian_K(X[idx_1],X[idx_2])*y[idx_2]

	L=0
	H=C
	alpha[idx_2]=-b/(2*a)
	if not (alpha[idx_2] >= L and alpha[idx_2] <= H):
		if alpha[idx_2] <= L:
			alpha[idx_2]=L
		elif alpha[idx_2] >= H:
			alpha[idx_2] = H

	alpha[idx_1]=(zeta-alpha[idx_2]*y[idx_2])*y[idx_1]
	return alpha

data=pd.read_csv('test.csv')



X=[np.array([data['height'][i],data['weight'][i]]) for i in range(len(data))]
y=[data['obese'][i] for i in range(len(data))]

for i in range(len(y)):
	if(y[i]==0):
		y[i]=-1

data_length=len(y)
alpha=[0 for i in range(data_length)]


for i in range(data_length-1):
	for j in range(i+1, data_length):
		optimize(X, y, data_length, i, j, alpha, 3)

print(alpha)

zeta=0
for i in range(data_length):
	zeta=zeta+y[i]*alpha[i]
print(zeta)

# ==========================================
# FULL VISUALIZATION (CS229 midpoint b)
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

X_np = np.array(X)
y_np = np.array(y)
alpha_np = np.array(alpha)

# ===== Compute w =====
w = np.zeros(X_np.shape[1])
for i in range(len(alpha_np)):
    w += alpha_np[i] * y_np[i] * X_np[i]

# ===== Compute b theo công thức hình m gửi =====
# b* = - ( max_{y=-1} w^T x + min_{y=1} w^T x ) / 2

wx = X_np @ w

max_neg = np.max(wx[y_np == -1])
min_pos = np.min(wx[y_np == 1])

b = - (max_neg + min_pos) / 2

print("w =", w)
print("b =", b)

# ==========================================
# START PLOTTING
# ==========================================

plt.figure(figsize=(8,6))

# Plot class +1
plt.scatter(
    X_np[y_np == 1][:,0],
    X_np[y_np == 1][:,1],
)

# Plot class -1
plt.scatter(
    X_np[y_np == -1][:,0],
    X_np[y_np == -1][:,1],
)

# Highlight support vectors
sv_mask = alpha_np > 1e-5
plt.scatter(
    X_np[sv_mask][:,0],
    X_np[sv_mask][:,1],
    s=150,
    facecolors='none',
)

# ==========================================
# Decision boundary + margins
# ==========================================

x_min, x_max = X_np[:,0].min(), X_np[:,0].max()
x_vals = np.linspace(x_min, x_max, 300)

if abs(w[1]) > 1e-8:

    # decision boundary
    y_vals = -(w[0]*x_vals + b) / w[1]
    plt.plot(x_vals, y_vals)

    # margin +1
    y_margin1 = -(w[0]*x_vals + b - 1) / w[1]
    plt.plot(x_vals, y_margin1, linestyle='--')

    # margin -1
    y_margin2 = -(w[0]*x_vals + b + 1) / w[1]
    plt.plot(x_vals, y_margin2, linestyle='--')

else:
    # vertical line case
    x_vertical = -b / w[0]
    plt.axvline(x=x_vertical)

plt.xlabel("height")
plt.ylabel("weight")
plt.title("SVM Decision Boundary (CS229 midpoint b)")
plt.show()