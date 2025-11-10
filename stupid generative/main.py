import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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
	

#perform shit with gaussian
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

#train model 
data=pd.read_csv('data.csv')
X_mean=np.array(
	[sum(data['x'])/len(data),sum(data['y'])/len(data)]
)

X=[
	np.array([data['x'][i],data['y'][i]])-X_mean
	for i in range(len(data))
]


number_of_labels=3

theta=[
	np.array([0,0])
	for i in range(number_of_labels)
]

for e in range(1000):
	for k in range(number_of_labels):
		val_der=softmax_der(X,data['label'],len(data),theta,k,number_of_labels)
		theta[k]=theta[k]+0.01*val_der
		
number_of_error=0
for i in range(len(data)):
	softmax_arr=[softmax(X[i],theta,k,number_of_labels) for k in range(number_of_labels)]
	if max(softmax_arr)!=softmax_arr[data['label'][i]]:
		number_of_error+=1

def calculate_step(sample_x,theta,chosen_label,number_of_labels,cov_matrices,x_mean):
	p_part=np.zeros(2)
	for j in range(number_of_labels):
		p_part=p_part+theta[j]*softmax(sample_x,theta,chosen_label,number_of_labels)
	p_part=theta[chosen_label]-p_part
	
	normal_part_div=0
	for j in range(number_of_labels):
		normal_part_div=normal_part_div+gaussian_dist(sample_x,X_labeled_mean[chosen_label],cov_matrices[chosen_label])
	normal_part_div=1.0
	
	normal_part=gaussian_grad(sample_x,X_labeled_mean[chosen_label],cov_matrices[chosen_label])
	
	return p_part+normal_part_div*normal_part

sample_x=np.array([9.42,0.32])
x_move=[sample_x]
for epoch in range(100):
	print(softmax(sample_x,theta,0,3))
	step=calculate_step(sample_x,theta,0,3,corvariance_matrices_labeled,X_labeled_mean)
	print(step)
	sample_x=sample_x+0.01*step 
	x_move.append(sample_x)

# as usual the visualize part was written by gpt cuz im too lazy
# Visualization với vùng màu dự đoán theo softmax và legend cho mỗi label
x_min, x_max = data['x'].min() - 1, data['x'].max() + 1
y_min, y_max = data['y'].min() - 1, data['y'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = np.zeros(xx.shape)
for i in range(xx.shape[0]):
	for j in range(xx.shape[1]):
		point = np.array([xx[i, j], yy[i, j]]) - X_mean
		softmax_vals = [softmax(point, theta, k, number_of_labels) for k in range(number_of_labels)]
		Z[i, j] = np.argmax(softmax_vals)

plt.figure(figsize=(8, 6))
contour = plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(number_of_labels + 1) - 0.5, cmap='viridis')

scatter = plt.scatter(data['x'], data['y'], c=data['label'], s=20, cmap='viridis', edgecolors='k', label='Data points')
plt.plot([x_move[i][0] for i in range(len(x_move))],
         [x_move[i][1] for i in range(len(x_move))],
         '-', c='red', label='Path')
plt.plot([x_move[0][0],x_move[-1][0]],[x_move[0][1],x_move[-1][1]],'x',c='red')

# Legend cho từng label
handles = []
for k in range(number_of_labels):
	handles.append(plt.Line2D([0], [0], marker='o', color='w',
	                          label=f'Label {k}',
	                          markerfacecolor=plt.cm.viridis(k / number_of_labels),
	                          markersize=8, markeredgecolor='k'))
handles.append(plt.Line2D([0], [0], marker='x', color='red', label='Path', markersize=8))
plt.legend(handles=handles, loc='upper right')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Softmax Decision Regions and Data Points')
plt.show()
