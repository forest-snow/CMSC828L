import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib.lines import Line2D

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
    # x_train = x_train.reshape(x_train.shape[0], n_pixels).astype('float32')
    # x_test = x_test.reshape(x_test.shape[0], n_pixels).astype('float32')
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # normalize
    x_train = x_train/255.0
    x_test = x_test/255.0

    return x_train, y_train, x_test, y_test

def simple_model():
    model = Sequential()
    model.add(Dense(n_pixels, input_dim=n_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_scores(scores):
    f = plt.figure(1)
    plt.plot(scores['acc'])
    plt.plot(scores['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper left')
    f.savefig('scores.png')

def plot_weights_biases(model):
    f = plt.figure(2)
    weights = []
    biases = []
    wb = []
    for i, matrix in enumerate(model.get_weights()):
        layer = int(i / 2) + 1
        max_val = np.max(matrix)
        min_val = np.min(matrix)
        mean_val = np.mean(matrix)
        if i % 2:
            wb.append(('Bias '+str(layer), max_val, 'c'))
            wb.append(('Bias '+str(layer), min_val, 'y'))
            wb.append(('Bias '+str(layer), mean_val, 'm'))

        else:
            wb.append(('Weight '+str(layer), max_val, 'c'))
            wb.append(('Weight '+str(layer), min_val, 'y'))
            wb.append(('Weight '+str(layer), mean_val, 'm'))

    labels = [v[0] for v in wb]
    values = [v[1] for v in wb]
    colors = [v[2] for v in wb]
    plt.scatter(labels, values, c=colors)

    plt.title('Analyzing weights and biases')
    plt.ylabel('Values')
    elements = \
        [Line2D([0], [0], marker='o', color='c', label='Maximum'),
        Line2D([0], [0], marker='o', color='y', label='Minimum'),
        Line2D([0], [0], marker='o', color='m', label='Mean')]
    plt.legend(handles=elements, loc='upper left')
    f.savefig('wb.png')

def find_errors(model, x_test, y_test, limit=10):
    with open('errors.txt', 'w') as f:
        pred = model.predict_classes(x_test).reshape((-1,))
        y = np.argmax(y_test, axis=1)
        errors = np.nonzero(pred != y)[0]
        errors = errors[: min(limit, len(errors))]
        for e in errors:
            print('Image {} should have label {} but predicted as {}'\
                .format(e, y[e], pred[e]), file=f)


if __name__ == '__main__':
    print('load data')
    x_train, y_train, x_test, y_test = load_data()
    print('build model')
    model = cnn_model()
    print('fit model')
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=100, verbose=2)
    plot_scores(history.history)
    plot_weights_biases(model)
    find_errors(model, x_test, y_test)
