import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression

df = pd.read_csv('reglin_3.csv')

X = df[['x']].values
y = df['y'].values

xl = np.linspace(-0.5, 15, 500)

lr = LinearRegression().fit(X, y)

yl = lr.predict(xl.reshape(-1, 1))

plt.plot(X, y, 'g.')
plt.plot(xl, yl, 'r-')
plt.show()

print(lr.score(X, y))
