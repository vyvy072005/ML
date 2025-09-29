import numpy as np
import pandas as pd

x1_num = np.linspace(-10, 10, 500)
x2_num = np.linspace(-1, 1, 500)


y = (np.tan(x1_num))/(3 + np.exp(-2 * x2_num))


data = {
    "x1": x1_num,
    "x2": x2_num,
    "y": y,
}

dataFm = pd.DataFrame(data)

dataFm.to_csv("generated.csv")
