from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

X = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([1,2,3])

df = pd.DataFrame(X, columns=['a','b','c'])

print(df.corr())

lr = LinearRegression()
lr.fit(X,y)

print(lr.coef_)

