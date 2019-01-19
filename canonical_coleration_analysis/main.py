import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import cv2
import numpy as np

# X = np.random.randn(3, 4)
# Y = np.random.randn(3, 5)

# cv2.imshow('Image 1', X)
# cv2.waitKey(30)

# cv2.imshow('Image 2', Y)
# cv2.waitKey(30)

X = cv2.imread('picture_1.png')
Y = cv2.imread('picture_2.jpg')

# X = np.array([np.array(cv2.imread('picture_1.png')) for i in range(len('picture_1.png'))])
# X_pixels = X.flatten().reshape(3, 2448, 3264)
# print X_pixels.shape

# Y = np.array([np.array(cv2.imread('picture_2.png')) for i in range(len('picture_2.png'))])
# Y_pixels = Y.flatten().reshape(3, 2448, 3264)
# print Y_pixels.shape

# X_resize = np.rollaxis(X, axis=2, start=-3)
# Y_resize = np.rollaxis(Y, axis=2, start=-3)

X = cv2.resize(X, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
Y = cv2.resize(Y, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

print('X_resize: ', X)
print('Y_resize: ', Y)

cv2.imshow('Predict', X)
cv2.waitKey(30)

# cca = CCA(n_components=1)
# cca.fit(X, Y)
#
# X_c, Y_c = cca.transform(X, Y)
#
# print('X_c: ', X_c)
# print('Y_c: ', Y_c)
#
# predict = cca.predict(X)
#
# print('predict: ', predict, type(predict))

# https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3
# Finding Correlation Between Many Variables (Multidimensional Dataset) with Python

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# data = pd.read_csv('https://www.dropbox.com/s/4jgheggd1dak5pw/data_visualization.csv?raw=1', index_col=0)
# corr = data.corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0, len(data.columns), 1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=90)
# ax.set_yticks(ticks)
# ax.set_xticklabels(data.columns)
# ax.set_yticklabels(data.columns)
# plt.show()

# from PIL import Image
# import numpy as np
# your_2d_array = np.random.randn(2000, 2000)
# img_array = []
# for x in your_2d_array.reshape(2000 * 2000):
#     if x == 1.0:
#         img_array.append((255, 0, 0)) # RED
#     elif x == 2.0:
#         img_array.append((0, 255, 0)) # GREEN
#     elif x == 3.0:
#         img_array.append((0, 0, 255)) # BLUE
#     elif x == 4.0:
#         img_array.append((0, 0, 0)) # BLACK
#     elif x == 5.0:
#         img_array.append((255, 255, 255)) # WHITE
#
# img = Image.new('RGB', (2000, 2000))
# img.putdata(img_array)
# img.show()
