import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def loss_function(w, b, x, y, number_of_case):
    res = 0.0
    for i in range(number_of_case):
        res += ((w * x[i] + b - y[i]) ** 2)
    res = res / number_of_case
    res = res * 0.5
    return res

def gradient_descent(w, b, x, y, number_of_case, learning_rate):
    derivative_w = 0
    derivative_b = 0

    for i in range(number_of_case):
        derivative_w += (w * x[i] + b - y[i]) * x[i]
        derivative_b += (w * x[i] + b - y[i])

    derivative_w /= number_of_case
    derivative_b /= number_of_case

    return [w - learning_rate * derivative_w, b - learning_rate * derivative_b]

data = pd.read_csv('test.csv')
number_of_case = len(data['x'])
w = 0
b = 0
x = data['x']
y = data['y']

print(loss_function(w, b, x, y, number_of_case))

learning_rate = 0.0001
for q in range(1000):
    [w, b] = gradient_descent(w, b, x, y, number_of_case, learning_rate)

print(loss_function(w, b, x, y, number_of_case))
print([w, b])

plt.scatter(x, y)
plt.plot(x, w * x + b, color='red')  
plt.show()
