import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

seed = 7
np.random.seed(seed)
n_pixels=784
n_classes=10 

def load_data():
    x_train = np.load('trainImages.npy')
    y_train = np.load('trainLabels.npy')
    x_test = np.load('testImages.npy')
    y_test = np.load('testLabels.npy')
    # reshape
    x_train = x_train.reshape(x_train.shape[0], n_pixels).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], n_pixels).astype('float32')

    # normalize
    x_train = x_train/255.0
    x_test = x_test/255.0

    return x_train, y_train, x_test, y_test

def build_model():
    model = Sequential()
    model.add(Dense(n_pixels, input_dim=n_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_scores(scores):
    plt.plot(scores['acc'])
    plt.plot(scores['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200, verbose=2)
    plot_scores(history.history)
