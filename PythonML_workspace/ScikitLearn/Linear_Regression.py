# 공부 시간에 따른 시험 점수 

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('LinearRegressionData.csv')
print(dataset.head())

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x, y)

reg = LinearRegression()
reg.fit(x, y) #학습

y_pred = reg.predict(x)
print(y_pred)

#데이터 시각화
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='green')
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')

# plt.show() # 그래프

print(reg.predict([[9]])) # 9시간 공부했을 경우 예상점수

print(reg.coef_) # 기울기
print(reg.intercept_) # y절편

# y = mx +b
X = 9
Y = reg.coef_ * X + reg.intercept_
print(Y)

# --------------------------------------------------------
# 데이터 세트 분리



