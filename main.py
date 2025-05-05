import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def loss_function(x,y,w,m,n):
	res=0
	for i in range(m):
		tmp=0
		for j in range(n):
			tmp=tmp+(x[i][j]*w[j]-y[i])**2
		res+=tmp 
	return res/(2*m)

def linear_regression_gradient_descent(X,y,theta,alpha) -> np.ndarray:    
	m, n = X.shape
	derivatives=np.zeros(n)
	for i in range(n):
		derivative=0
		for j in range(m):
			derivative=derivative+(np.dot(theta,X[j])-y[j])*X[j][i]
		derivative/=m
		derivatives[i]=derivative
	theta=theta-derivatives*alpha
	return theta
	
data=pd.read_csv("test.csv").to_dict()
number_of_test=len(data['x'])

x=[]
y=[]
for q in range(number_of_test):
	x.append([])
	for j in data:
		if j != "y":
			x[q].append(data[j][q])
		else:
			y.append(data[j][q])
x=np.array(x)
y=np.array(y)
w=np.dot(
	np.linalg.pinv(
		np.dot(
			np.transpose(x),
			x
		)
	),
	np.dot(
		np.transpose(x),
		y 
	)
)
plt.scatter(x,y)
plt.plot(x, w * x, color='red')  
plt.show()