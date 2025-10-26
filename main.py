import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt



def sigmoid(w,x):
	res=1/(1+math.exp(-w@x))
	return res

def gradient_ascent(w,x,y,alpha):
    m,n=x.shape
    derivatives=[]
    for j in range(n):
        derivative=0
        for i in range(m):
            derivative=derivative+x[i][j]*(y[i]-sigmoid(w,x[i]))
        derivatives.append(derivative)
    derivatives=np.array(derivatives)
    w=w+alpha*derivatives
    return w

def feature_scale(x,max_val,min_val):
    return (x-min_val)/(max_val-min_val)

def likelihood(w,x,y):
    m,n=x.shape
    res=1
    for i in range(m):
        prediction=sigmoid(w,x[i])
        res=res*(prediction**y[i])*((1-prediction)**(1-y[i]))
    return res

data=pd.read_csv('test.csv').to_dict()
x=[]
y=[]
max_height=max(data['height'])
min_height=min(data['height'])
max_weight=max(data['weight'])
min_weight=min(data['weight'])

for i in range(len(data['obese'])):
    x.append([(data['height'][i]-min_height)/(max_height-min_height),(data['weight'][i]-min_weight)/(max_weight-min_weight)])
    y.append(data['obese'][i])

iter_time=10000
w=np.array([-9.91594276, 20.73774018])

likelihoods = []
for q in range(iter_time):
    w = gradient_ascent(w, np.array(x), np.array(y), 0.003)
sample=[feature_scale(178,max_height,min_height),feature_scale(85,max_weight,min_weight)]
print(likelihood(w,np.array(x),np.array(y)))