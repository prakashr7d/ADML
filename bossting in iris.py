import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

df = pd.DataFrame(load_iris()['data'],
                  columns=load_iris()['feature_names'])
df['target'] = load_iris()['target']

X = df.drop(['target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

ada = AdaBoostClassifier(base_estimator=dtc, n_estimators=5, learning_rate=0.1)
ada.fit(X_train, y_train)

gbd = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1)
gbd.fit(X_train, y_train)

dtc_score = dtc.score(X_test, y_test)
ada_score = ada.score(X_test, y_test)
gbd_score = gbd.score(X_test, y_test)

print("--- score(accuracy) ---")
print('Decision Tree: {:.2f}'.format(dtc_score))
print('AdaBoost: {:.2f}'.format(ada_score))
print('Gradient Boost: {:.2f}'.format(gbd_score))

# k-fold cross validation 
print("---- cross-val results ----")
dtc_cv = cross_val_score(dtc, X_train, y_train, cv=4)
print(f"decission tree classier: {dtc_cv.mean()}")
ada_cv = cross_val_score(ada, X_train, y_train, cv=4)
print(f"ada boost classier: {ada_cv.mean()}")
gbd_cv = cross_val_score(gbd, X_train, y_train, cv=4)
print(f"gradient boost classier: {gbd_cv.mean()}")

dtc_cv = cross_val_score(dtc, X_test, y_test, cv=4)
print(f"decission tree classier: {dtc_cv.mean()}")
ada_cv = cross_val_score(ada, X_test, y_test, cv=4)
print(f"ada boost classier: {ada_cv.mean()}")
gbd_cv = cross_val_score(gbd, X_test, y_test, cv=4)
print(f"gradient boost classier: {gbd_cv.mean()}")

print("---- kfold result ----")

dtc_cv_kfold = cross_val_score(dtc, X_train, y_train, cv=KFold(n_splits=5))
print(f"decission tree classifier: {dtc_cv_kfold.mean()}")
ada_cv_kfold = cross_val_score(ada, X_train, y_train, cv=KFold(n_splits=5))
print(f"ada boost classifier: {ada_cv_kfold.mean()}")
gbd_cv_kfold = cross_val_score(gbd, X_train, y_train, cv=KFold(n_splits=5))
print(f"gradient boost classifier: {gbd_cv_kfold.mean()}")


print(f"decission tree classifier: {dtc_cv_kfold.mean()}")
ada_cv_kfold = cross_val_score(ada, X_test, y_test, cv=KFold(n_splits=5))
print(f"ada boost classifier: {ada_cv_kfold.mean()}")
gbd_cv_kfold = cross_val_score(gbd, X_test, y_test, cv=KFold(n_splits=5))
print(f"gradient boost classifier: {gbd_cv_kfold.mean()}")

print("---- stratified kfold ----")

Stratified_Kfold = cross_val_score(dtc, X_train, y_train, cv=StratifiedKFold(n_splits=5))
print(f"decission tree classifier: {Stratified_Kfold.mean()}")
Stratified_Kfold = cross_val_score(ada, X_train, y_train, cv=StratifiedKFold(n_splits=5))
print(f"ada boost classifier: {Stratified_Kfold.mean()}")
Stratified_Kfold = cross_val_score(gbd, X_train, y_train, cv=StratifiedKFold(n_splits=5))
print(f"gradient boost classifier: {Stratified_Kfold.mean()}")


print(f"decission tree classifier: {Stratified_Kfold.mean()}")
Stratified_Kfold = cross_val_score(ada, X_test, y_test, cv=StratifiedKFold(n_splits=5))
print(f"ada boost classifier: {Stratified_Kfold.mean()}")
Stratified_Kfold = cross_val_score(gbd, X_test, y_test, cv=StratifiedKFold(n_splits=5))
print(f"gradient boost classifier: {Stratified_Kfold.mean()}")
