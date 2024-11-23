import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set()

df = pd.read_csv('iris.csv')
df.info()

print(df['species'].value_counts())

print(df)

sns.pairplot(df, hue='species')
plt.show()

# setosa and versicolor

Xsv = df.iloc[:100, 0:4]  # setosa, versicolor
ysv = df.iloc[:100, 4]

Xvi = df.iloc[50:150, 0:4]  # versicolor, virginica
yvi = df.iloc[50:150, 4]
yvi = np.where(yvi == 'versicolor', -1, 1)

Xsi = pd.concat([df.iloc[:50, 0:4], df.iloc[100:150, 0:4]])  # setosa, virginica
ysi = pd.concat([df.iloc[:50, 4], df.iloc[100:150, 4]])


from sklearn.linear_model import Perceptron

pp1 = Perceptron().fit(Xsv, ysv)
print(pp1.score(Xsv, ysv))

y_pred = pp1.predict(Xsv)


from sklearn.preprocessing import StandardScaler

# standaryzacja danych
sc = StandardScaler()
X_std = sc.fit_transform(Xvi);

from sklearn.linear_model import Ridge

ada = Ridge(alpha=0.0).fit(X_std, yvi)
print(ada.score(X_std, yvi))

# X_train, X_test, y_train, y_test, = train_test_split(X_std, yvi, test_size=0.3, stratify=ysi)
# X_train.shape,

# pp2 = Perceptron().fit(X_train, y_train)
# train_score =