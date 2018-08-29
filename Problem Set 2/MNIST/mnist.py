#dataset from http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
#requires around 4GB of CPU RAM

import numpy as np
import keras
from sklearn.utils import shuffle

np.random.seed(1984)

trainImages = np.loadtxt(open('train_images_mnist.csv', 'rb'), delimiter = ',', skiprows = 0)
testImages = np.loadtxt(open('test_images_mnist.csv', 'rb'), delimiter = ',', skiprows = 0)
trainLabels = np.loadtxt(open('train_labels_mnist.csv', 'rb'), delimiter = ',', skiprows = 0, dtype = 'uint8')
testLabels = np.loadtxt(open('test_labels_mnist.csv', 'rb'), delimiter = ',', skiprows = 0, dtype = 'uint8')

labels = np.append(testLabels, trainLabels)
images = np.append(testImages, trainImages, axis = 0)

images=np.multiply(images, 256)

images=images.astype('uint8')

labels = shuffle(labels)
images = shuffle(images)

labels = keras.utils.to_categorical(labels)

np.savetxt('mnistLabels.csv', labels, delimiter = ',', fmt = '%u')
np.savetxt('mnistImages.csv', images, delimiter = ',', fmt = '%u')