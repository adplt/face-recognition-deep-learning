import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

X = np.random.randn(3, 4)
Y = np.random.randn(3, 5)

print('X: ', X)
print('Y: ', Y)

cca = CCA(n_components=1)
cca.fit(X, Y)

# X_c, Y_c = cca.transform(X, Y)

# print('X_c: ', X_c)
# print('Y_c: ', Y_c)

predict = cca.predict(X)

print('predict: ', predict, type(predict))

# https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3
# Finding Correlation Between Many Variables (Multidimensional Dataset) with Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('https://www.dropbox.com/s/4jgheggd1dak5pw/data_visualization.csv?raw=1', index_col=0)
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(data.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
