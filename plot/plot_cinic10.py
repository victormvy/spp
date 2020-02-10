import keras
import matplotlib.pyplot as plt
import numpy as np
import random
from dataset2 import Dataset

ds = Dataset('cinic10')

gen = ds.generate_train(128, {}).__iter__()

fig=plt.figure(figsize=(8, 8))
columns = 8
rows = 3
plt.axis('off')
for i in range(1, columns*rows +1):
    x_train, y_train = next(gen)
    img = x_train[random.randint(0, x_train.shape[0])]
    ax = fig.add_subplot(rows, columns, i)
    ax.axis('off')
    plt.imshow(img, cmap='brg')
plt.show()