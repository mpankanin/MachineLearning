import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

df1 = pd.read_csv('reglin_1.csv')
df1.info()

plt.plot(df1['x'], df1['y'], 'g.')
plt.show()

# y = ax + b
# a - slope
# b - intercept

# a = sum((xi - xm)*(yi - ym)) / sum(pow(x1 - xm))
# b = ym - a * xm

xm = df1['x'].mean()  # x mean value
ym = df1['y'].mean()  # y mean value
xc = df1['x'] - xm  # x centered values (the mean of those values is equals to 0)
yc = df1['y'] - ym  # y centered values (the mean of those values is equals to 0)

a = (xc*yc).sum() / np.square(xc).sum()
b = ym - (a * xm)
print(a, b)

x1 = np.linspace(-0.5, 10.5, 500)  #generating 500 points from -0.5 to 10.5 range
y1 = a*x1 + b  #calculating y values for generated x values
plt.plot(df1['x'], df1['y'], 'g.')
plt.plot(x1, y1, 'r')

# Calculate linear regression score

tss = np.square(df1['y'] - ym).sum()
rss = np.square(df1['y'] - (a*df1['x'] + b)).sum()
rmse = np.sqrt(rss / len(df1))
r2 = 1 - rss/tss
print(rmse, r2)

# Calculation using sklearn library

from sklearn.linear_model import LinearRegression





