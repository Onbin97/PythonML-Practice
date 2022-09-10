# Polynomial Regression(다항 회귀)

from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


dataset = pd.read_csv('PolynomialRegressionData.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 단순 선형회귀

reg = LinearRegression()
reg.fit(X, y)

plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, reg.predict(X), color='green') # 선 그래프
plt.title('Score by hours (genius)')
plt.xlabel('hours')
plt.ylabel('score')

# plt.show()

print(reg.score(X, y))

# 다항 회귀

poly_reg = PolynomialFeatures(degree=4) # 2차
X_poly = poly_reg.fit_transform(X) 

print(X_poly[:5]) # x^0, x^1, x^2

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# plt.scatter(X, y, color='blue') # 산점도
# plt.plot(X, lin_reg.predict(X_poly), color='green') # 선 그래프
# plt.title('Score by hours (genius)')
# plt.xlabel('hours')
# plt.ylabel('score')

# plt.show()

X_range = np.arange(min(X), max(X), 0.1) # X 의 최소값에서 최대값까지의 범위를 0.1 단위로 잘라서 데이터를 생성
X_range = X_range.reshape(-1, 1) # 컬럼한개만큼(1) 로우는 자동으로(-1)

plt.scatter(X, y, color='blue') # 산점도
plt.plot(X_range, lin_reg.predict(poly_reg.fit_transform(X_range)), color='green') # 선 그래프
plt.title('Score by hours (genius)')
plt.xlabel('hours')
plt.ylabel('score')

# plt.show()

# 2시간 공부 후 시험 성적은 ??

print(reg.predict([[2]]))
print(lin_reg.predict(poly_reg.fit_transform([[2]])))

print(reg.score(X, y))
print(lin_reg.score(X_poly, y))