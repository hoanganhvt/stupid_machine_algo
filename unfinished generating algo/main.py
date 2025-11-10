import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt


def softmax(x,theta,chosen_label,number_of_labels):
	result=0.0
	for j in range(number_of_labels):
		result=result+np.exp(theta[j]@x)
	result=np.exp(theta[chosen_label]@x)/result
	return result


def softmax_der(X,y,number_of_samples,theta,chosen_label,number_of_labels):
	result=np.array([0 for i in range(len(X[0]))])
	for i in range(number_of_samples):
		result=result+(float(y[i]==chosen_label)-softmax(X[i],theta,chosen_label,number_of_labels))*X[i]
	return result


def gaussian_dist(x, mean, cov):
	d=len(x)
	det=np.linalg.det(cov)
	inv=np.linalg.inv(cov)
	norm_const=1.0/((2*np.pi)**(d/2)*np.sqrt(det))
	diff=x-mean
	exponent=-0.5*diff.T@inv@diff
	return norm_const*np.exp(exponent)


def gaussian_grad(x, mean, cov):
	d=len(x)
	inv=np.linalg.inv(cov)
	diff=(x-mean).reshape(d,1)
	p=gaussian_dist(x,mean,cov)
	grad=-p*(inv@diff).reshape(d)
	return grad


data=pd.read_csv('data.csv')
X=[np.array([data['x'][i],data['y'][i]]) for i in range(len(data))]

number_of_labels=3
X_labeled=[
	[] for i in range(number_of_labels)
]

X_labeled_mean=[
	np.zeros(2) for i in range(number_of_labels)
]

#group point by label
for label in range(number_of_labels):
	for i in range(len(X)):
		if data['label'][i]==label:
			X_labeled[label].append(X[i])

#calculate mean of each label
for label in range(number_of_labels):
	for i in range(len(X_labeled[label])):
		X_labeled_mean[label]+=X_labeled[label][i]
	X_labeled_mean[label]/=len(X_labeled[label])


# Calculate the covariance matrices
corvariance_matrices_labeled=[
	np.cov(X_labeled[i],rowvar=False)
	for i in range(number_of_labels)
]

#train the stupid model 
X=[np.array([data['x'][i],data['y'][i],1]) for i in range(len(data))]
number_of_labels=3

theta=[
	np.array([0,0,0])
	for i in range(number_of_labels)
]

for e in range(1000):
	for k in range(number_of_labels):
		val_der=softmax_der(X,data['label'],len(data),theta,k,number_of_labels)
		theta[k]=theta[k]+0.01*val_der
		
print(theta)
def calculate_step(sample_x,theta,chosen_label,number_of_labels,cov_matrices,x_mean):
	p_part=np.zeros(3)
	for j in range(number_of_labels):
		p_part=p_part+theta[j]*softmax(np.append(sample_x,1),theta,chosen_label,number_of_labels)
	p_part=theta[chosen_label]-p_part
	
	normal_part_div=0
	for j in range(number_of_labels):
		normal_part_div=normal_part_div+gaussian_dist(sample_x,X_labeled_mean[chosen_label],cov_matrices[chosen_label])
	normal_part_div=1.0
	
	normal_part=gaussian_grad(sample_x,X_labeled_mean[chosen_label],cov_matrices[chosen_label])
	
	return np.delete(p_part,-1)+normal_part_div*normal_part

sample_x=np.array([9.42,0.32])

for epoch in range(150):
	print(softmax(np.append(sample_x,1),theta,0,3))
	step=calculate_step(sample_x,theta,0,3,corvariance_matrices_labeled,X_labeled_mean)
	print(step)
	sample_x=sample_x+0.1*step 
	