import pandas as pd
import math
import random

df= pd.read_csv("MCDO/breast-cancer.csv")
y=df["diagnosis"]
y = y.map({'M': 1, 'B': 0})

x = df.iloc[:,2:]
x = (x - x.mean()) / x.std()

def sigmoide(z):
    return 1 / (1 + math.exp(-z)) 

x_sig = x.applymap(sigmoide)
inicial = [random.uniform(-1, 1) for _ in range(x.shape[1])]

#print(y)
print(x_sig)

