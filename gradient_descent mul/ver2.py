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
	derivative_b=0.0
	for i in range(n):
		err=np.dot(w,x[i])+b-y[i]
		derivative_b+=err
		for j in range(w_len):
			derivative_w[j]=derivative_w[j]+err*x[i,j]
	derivative_w=derivative_w/n 
	w=w-L*derivative_w
	b=b-L*derivative_b
	return [w,b]

b_init = 0
w_init = np.array([ 0, 0, 0, 0])
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
for i in range(1000):
	[w,b]=gradient_descent(w_init,b_init,x_train,y_train,0.00001)
print(w,' ',b)
print(loss_function(w_init,b_init,x_train,y_train))