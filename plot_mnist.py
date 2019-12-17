import keras
import matplotlib.pyplot as plt
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

fig=plt.figure(figsize=(8, 8))
columns = 7
rows = 3
plt.axis('off')
for i in range(1, columns*rows +1):
    img = x_train[random.randint(0, x_train.shape[0])]
    ax = fig.add_subplot(rows, columns, i)
    ax.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()