#from https://www.kaggle.com/zalando-research/fashionmnist (using .csv files)

import numpy as np
from sklearn.utils import shuffle
import keras

np.random.seed(451)

train = np.loadtxt(open('fashion-mnist_train.csv', 'rb'), delimiter = ',', skiprows = 1, dtype = 'uint8')
test = np.loadtxt(open('fashion-mnist_test.csv', 'rb'), delimiter = ',', skiprows = 1, dtype = 'uint8')

data = np.append(test, train, axis = 0)

data=shuffle(data)

labels = data[:, 0]

labels = keras.utils.to_categorical(labels)

np.delete(data, 0, 1)

np.savetxt('fashionMnistLabels.csv', labels, delimiter = ',', fmt = '%u')
np.savetxt('fashionMnistImages.csv', data, delimiter = ',', fmt = '%u')