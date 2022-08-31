# 경사 하강법
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import SGDRegressor # SGD : Stochastic Gradient Descent(확률적 경사 하강법)
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('LinearRegressionData.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 훈련 80 : 테스트 20

print(X, len(X))
print(X_train, len(X_train))
print(X_test, len(X_test))

sr = SGDRegressor()
sr.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, sr.predict(X_train), color='green')
plt.title('Score by hours(Train data, SGD)')
plt.xlabel('hours')
plt.ylabel('score')

plt.show()