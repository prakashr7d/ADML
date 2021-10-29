from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = pd.Series(cancer.target)

X = df.drop(['target'], axis=1)
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

lr = LogisticRegression()
lr.fit(X_train, y_train)

bc = BaggingClassifier(lr, n_estimators=100, max_features=10, max_samples=100, n_jobs=-1)
bc.fit(X_train, y_train)


train_acc_lr = lr.score(X_train, y_train)
train_acc_bc = bc.score(X_train, y_train)
print("training accuracy of LR: ", train_acc_lr)
print("training accuracy of BC: ", train_acc_bc)
lr_score = lr.score(X_test, y_test)
bc_score = bc.score(X_test, y_test)
print("testing accuracy of LR:", lr_score)
print("testing accuracy of BC:", bc_score)


print("Difference in accuracy between logistic regression and the"
      f" bagging classifier(with logistic regression as a base model) is: {lr_score - bc_score }")

