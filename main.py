import pandas as pd
import matplotlib.pyplot as plt


def loss_function(w,b,x,y):
	result=0
	for i in range(len(x)):
		result+=(w*x[i]+b-y[i])
	return result

def gradient_descent(w,b,x,y,learning_rate,n):
	derivative_w=0
	derivative_b=0

	for i in range(n):
		derivative_w+=(w*x[i]-y[i])*x[i]
		derivative_b+=(w*x[i]-y[i])
	derivative_w=derivative_w/float(n)
	derivative_b=derivative_b/float(n)
	w-=learning_rate*derivative_w
	b-=learning_rate*derivative_b
	return [w,b]

data=pd.read_csv('test.csv').to_dict()
w=0
b=0
x=[]
y=[]
L=0.0000001


for i in data['x']:
	x.append(data['x'][i])
for i in data['y']:
	y.append(data['y'][i])

for i in range(100000):
	[w,b]=gradient_descent(w,b,x,y,L,len(x))
plt.scatter(x,y)
plt.plot(list(range(0,100)),[w*x+b for x in range(0,100)],color="red")
plt.show()