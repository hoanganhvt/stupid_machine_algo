import numpy as np
import pandas as pd 

data=pd.read_csv('data.csv')
arr=np.array([1,2,3])

print(arr@arr)