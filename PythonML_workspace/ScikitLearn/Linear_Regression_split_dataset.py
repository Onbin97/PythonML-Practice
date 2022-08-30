# 훈련 데이터와 테스트데이터 분리

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('LinearRegressionData.csv')
print(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 훈련 80 : 테스트 20

print(X, len(X))
print(X_train, len(X_train))
print(X_test, len(X_test))

reg = LinearRegression()
reg.fit(X_train, y_train)

# 데이터 시각화
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Score by hours(Train data)')
plt.xlabel('hours')
plt.ylabel('score')

# plt.show()

# 데이터 시각화(Test set)
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Score by hours(Test data)')
plt.xlabel('hours')
plt.ylabel('score')

# plt.show()

print(reg.coef_, reg.intercept_)


#모델 평가

print(f'Score : {reg.score(X_test, y_test)}')