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
        if i % 2:
            val = np.max(matrix)
            wb.append(('Bias '+str(layer), val))
        else:
            val = np.max(np.mean(matrix, axis=0))
            wb.append(('Weight '+str(layer), val))
    labels = [v[0] for v in wb]
    values = [v[1] for v in wb]
    plt.scatter(labels, values)

    plt.title('Analyzing weights and biases')
    plt.ylabel('Maximum value')
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
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=2000, verbose=2)
    plot_scores(history.history)
    plot_weights_biases(model)
    find_errors(model, x_test, y_test)
