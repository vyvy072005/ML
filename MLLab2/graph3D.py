import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv('generated.csv').dropna()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["x1"], df["x2"], df["y"])

ax.plot(df["x1"], df["x2"], df["y"])


plt.show()
