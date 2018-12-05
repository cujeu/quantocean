import numpy as np 
import matplotlib.pyplot as plt

values= np.random.normal(90,2, 10000)
plt.hist(values,50)
plt.show()
