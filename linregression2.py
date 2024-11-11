import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression


df = pd.read_csv('reglin_2.csv')

X = df[['x']].values
y = df['y'].values

print(X, y)

lr = LinearRegression().fit(X, y)

xl = np.linspace(-0.5, 10.5, 500)

yl = lr.predict(xl.reshape(-1, 1))

plt.plot(X[:,0], y, 'g.')
plt.plot(xl, yl, 'r-')

plt.show()

print(lr.score(X, y))
