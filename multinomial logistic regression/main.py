import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import math


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


data=pd.read_csv('data.csv')
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
		
number_of_error=0
for i in range(len(data)):
	softmax_arr=[softmax(X[i],theta,k,number_of_labels) for k in range(number_of_labels)]
	if max(softmax_arr)!=softmax_arr[data['label'][i]]:
		number_of_error+=1

print(number_of_error)

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import math


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


data=pd.read_csv('data.csv')
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
		
number_of_error=0
for i in range(len(data)):
	softmax_arr=[softmax(X[i],theta,k,number_of_labels) for k in range(number_of_labels)]
	if max(softmax_arr)!=softmax_arr[data['label'][i]]:
		number_of_error+=1

print(number_of_error)

#too lazy to write so this part is written by claude :)

# ============ PHẦN VISUALIZATION THÊM VÀO ============

# Tính predictions
predictions = []
for i in range(len(data)):
	softmax_arr=[softmax(X[i],theta,k,number_of_labels) for k in range(number_of_labels)]
	predicted_label = np.argmax(softmax_arr)
	predictions.append(predicted_label)

# Tạo mesh grid cho decision boundaries
x_min, x_max = data['x'].min() - 1, data['x'].max() + 1
y_min, y_max = data['y'].min() - 1, data['y'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
					 np.linspace(y_min, y_max, 200))

# Dự đoán cho mỗi điểm trên mesh
Z = []
for i in range(len(xx.ravel())):
	point = np.array([xx.ravel()[i], yy.ravel()[i], 1])
	softmax_arr = [softmax(point, theta, k, number_of_labels) for k in range(number_of_labels)]
	Z.append(np.argmax(softmax_arr))
Z = np.array(Z).reshape(xx.shape)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Dữ liệu gốc với nhãn thật
colors_true = ['red', 'blue', 'green']
for label in range(number_of_labels):
	mask = data['label'] == label
	axes[0].scatter(data['x'][mask], data['y'][mask], 
					c=colors_true[label], label=f'Label {label}', 
					alpha=0.8, s=80, edgecolors='black', linewidths=1.5)
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('Y', fontsize=12)
axes[0].set_title('Dữ liệu gốc (Nhãn thật)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Decision boundaries với dữ liệu dự đoán
from matplotlib.colors import ListedColormap
cmap_background = ListedColormap(['#ffcccc', '#ccccff', '#ccffcc'])
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background, levels=np.arange(number_of_labels+1)-0.5)
axes[1].contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.3, levels=np.arange(number_of_labels+1)-0.5)

# Vẽ điểm dữ liệu
predictions_array = np.array(predictions)
colors_pred = ['red', 'blue', 'green']
for label in range(number_of_labels):
	mask = predictions_array == label
	axes[1].scatter(data['x'][mask], data['y'][mask], 
					c=colors_pred[label], label=f'Predicted {label}', 
					alpha=0.8, s=80, edgecolors='black', linewidths=1.5)

# Đánh dấu các điểm dự đoán sai
errors_mask = predictions_array != data['label'].values
if errors_mask.sum() > 0:
	axes[1].scatter(data['x'][errors_mask], data['y'][errors_mask], 
					facecolors='none', edgecolors='orange', 
					s=250, linewidths=3, label='Lỗi dự đoán', marker='o')

axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('Y', fontsize=12)
axes[1].set_title(f'Vùng dự đoán và Decision Boundaries (Accuracy: {(len(data)-number_of_error)/len(data)*100:.1f}%)', 
				  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()