import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()


df = pd.read_csv('generated.csv')

# x2 - const

random_x2 = df['x2'].sample(n=1).iloc[0]


y = (np.tan(df.x1))/(3 + np.exp(-2 * random_x2))

plt.figure()
plt.plot(df['x1'], y, marker='o', linestyle='--', color='green') # График y(x2)
#plt.xlim(2, 8)
plt.xlabel('x2')
plt.ylabel('y')
plt.title(f'График y(x1) при x2 = {random_x2}')
plt.grid(True)
plt.show()


# x1 - const

random_x1 = df['x1'].sample(n=1).iloc[0]


y = (np.tan(random_x1))/(3 + np.exp(-2 * df.x2))



plt.figure()
plt.plot(df['x2'], y, marker='o', linestyle='-', color='green') # График y(x2)
plt.xlabel('x2')
plt.ylabel('y')
plt.title(f'График y(x2) при x1 = {random_x1}')
plt.grid(True)
plt.show()


print("Средние значения")
print(df.mean()[1:])
print("Минимальные значения")
print(df.min()[1:])
print("Максимальные значения")
print(df.max()[1:])


df_new = df[(df.x1 < df.x1.mean()) | (df.x2 < df.x2.mean())]
df_new.to_csv('new_csv', index=False)



