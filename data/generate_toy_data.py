import numpy as np
import pandas as pd

x = list(range(1, 7000))
y = list()

for item in x:
    if item % 7 == 0:
        y.append(1)
    else:
        y.append(0)

df = pd.DataFrame(list(zip(x, y)), columns=["x", "y"])
df.to_csv("./toy_dataset.csv", index=False)