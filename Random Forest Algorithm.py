import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./petrol_consumption.csv')
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
print(df)

y = df['Petrol_Consumption']
X = df.drop(['Petrol_Consumption'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
print(X_train)


def dec_tree(n_estim=20):
    rfc = RandomForestRegressor(n_estimators=n_estim)
    rfc.fit(X_train, y_train)

    prediction = rfc.predict(X_test)

    mae = mean_absolute_error(prediction, y_test)
    mse = mean_squared_error(prediction, y_test)
    rmse = mean_squared_error(prediction, y_test, squared=False)

    print(f"mean_squared_error: {mse}")
    print(f"mean_absolute_error: {mae}")
    print(f"root_mean_squared_error: {rmse}")


print("---estimators : 20----")

dec_tree(20)
print("---estimators: 50----")
dec_tree(50)
