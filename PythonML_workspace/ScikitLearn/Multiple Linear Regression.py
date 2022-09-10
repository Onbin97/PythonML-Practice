# One hot encoding

from re import L
import pandas as pd

from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_csv('MultipleLinearRegressionData.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(X, y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')

X = ct.fit_transform(X)

print(X)
# 1.0 = Home
# 0.1 = Library
# 0.0 = Cafe


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 훈련 80 : 테스트 20

# 다중 선형회귀 학습
reg = LinearRegression()
reg.fit(X_train, y_train)

# 예측값과 실제값 비교

y_pred = reg.predict(X_test)
print(y_pred)

print(y_test)

print(reg.coef_) # cafe가 공부가 가장 잘됨.
print(reg.intercept_)

print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))

# 다양한 평가 지표 (회귀 모델)
# MAE (Mean Absolute Error)     : (실제 값과 예측 값) 차이의 절대값
# MSE (Mean Squared Error)      : 차이의 제곱
# RMSE (Root Mean Squared Error): 차이의 제곱의 루트
# R2                            : 결정 계수 (1에 가까울수록 잘만들어진 회귀 모델)

# MAE
print(mean_absolute_error(y_test, y_pred))

# MSE
print(mean_squared_error(y_test, y_pred))

# RMSE
print(mean_squared_error(y_test, y_pred, squared=False))

# R2
print(r2_score(y_test, y_pred))