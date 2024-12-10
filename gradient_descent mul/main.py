import pandas as pd
import numpy as np

def loss_function(w,b,x,y):
	n=x.shape[0]
	result=0.0
	for i in range(n):
		tmp=(np.dot(x[i],w)+b-y[i])**2
		result=tmp
	result=result/(2*n)
	return tmp

def gradient_descent(w,b,x,y,L):
	[n,w_len]=x.shape
	derivative_w=np.zeros(w_len,)
	derivative_b=0

	for i in range(w_len):
		for j in range(n):
			derivative_w[i]=derivative_w[i]+(np.dot(w,x[j])+b-y[j])*x[j][i]
	derivative_w=derivative_w/n
	b=b-L*derivative_b
	return [w,b]

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

for i in range(1000):
	[w,b]=gradient_descent(w_init,b_init,x_train,y_train,0.001)
print(loss_function(w_init,b_init,x_train,y_train))