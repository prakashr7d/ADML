import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = pd.Series(cancer.target)

X = df.drop(['target'], axis=1)
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
print(f"shape before doing PCA: {X.shape}")
pca = PCA(n_components=2)
X = pca.fit_transform(X)
features = X.T
covariance_matrix = np.cov(features)
print("Covariance matrix: \n", covariance_matrix)

X

print(f"shape after doing PCA: {X.shape}")
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

lr = LogisticRegression()
lr.fit(X_train, y_train)

prediction = lr.predict(X_test)
mse = mean_squared_error(prediction, y_test)
print(f"mean squared error of logistic regression using PCA is : {mse}")
