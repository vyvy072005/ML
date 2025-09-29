import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

x1 = np.linspace(-100, 100, 3)



x2 = 4 + 4 * x1

x3 = (31+x1)//4
x4 = 29-4*x1
x5 = x1-1



plt.figure()
plt.plot(x2, x1, marker=' ', linestyle='-', color='black') # График y(x2)
plt.plot(x3, x1, marker=' ', linestyle='-', color='black') # График y(x2)
plt.plot(x4, x1, marker=' ', linestyle='-', color='black')
plt.plot(x5, x1, marker=' ', linestyle='-', color='black')



plt.xlim(-10, 20)
plt.ylim(-10,30)
plt.xlabel('x2')
plt.ylabel('x1')
plt.grid(True)
plt.show()


