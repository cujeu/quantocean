import numpy as np
from sklearn.metrics import accuracy_score

y_pred = [0, 2, 1, 3, 2,3]
y_true = [0, 1, 2, 3, 4,5]
print(accuracy_score(y_true, y_pred))

print(accuracy_score(y_true, y_pred, normalize=False))

a = np.arange(24).reshape((3, 4,2))
print(a)

arr = np.array(['1,233','2,343','3,443','3,434'])
arr = arr.astype('int32') 
  
# Print the array after changing 
# the data type 
print(arr) 
  
# Also print the data type 
print(arr.dtype) 


list1 = ['f', 'o', 'o']
list2 = ['hello', 'world']
result = [None]*(len(list1)+len(list2))
result[::2] = list1
result[1::2] = list2

print(result)
