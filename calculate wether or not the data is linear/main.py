import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 

def get_weight(central_x,tau,x_val):
   res=math.exp(
      -(x_val-central_x)**2/(tau**2)/2
   )
   return res 
   
def linear_regression(central_x,tau,x,y):
   weights=[[0.0 for j in range(len(x))] for i in range(len(x))]
   for i in range(len(x)):
      weights[i][i]=get_weight(central_x,tau,x[i][0])
   weights=np.array(weights)
   theta=np.linalg.pinv(
      np.transpose(x)@weights@x
   )@np.transpose(x)@weights@y
   return theta

data=pd.read_csv('data.csv')
x=[]
for i in range(len(data['x'])):
   x.append([data['x'][i],1.0])
x=np.array(x)
y=np.array(data['y'])

theta_list=[]
tau=0.5
for i in range(len(x)):
   theta=linear_regression(x[i][0],tau,x,y)
   theta_list.append(theta)

degree_of_data=1
current_sign=theta_list[0][0]/abs(theta_list[0][0])

for theta in theta_list:
   cur_theta=theta[0]
   if theta[0]==0:
      continue
   cur_theta_sign=cur_theta/abs(cur_theta)
   
   if cur_theta_sign != current_sign:
      degree_of_data+=1
      current_sign=cur_theta_sign


print(degree_of_data)