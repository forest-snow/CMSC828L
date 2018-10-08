import numpy as np
# import matplotlib
# matplotlib.use('agg') 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from matplotlib.lines import Line2D

# seed = 7
# np.random.seed(seed)
n_feat=9
n_classes=2 

def load_data(split=0.15):
    x = np.loadtxt('breastCancerData.csv', delimiter=',')
    y = np.loadtxt('breastCancerLabels.csv')
    y = to_categorical(y)


    size = x.shape[0]
    test = int(split*size)
    ind = np.random.permutation(size)
    train_ind = ind[:-test]
    test_ind = ind[-test:]
    x_train = x[train_ind]
    y_train = y[train_ind]
    x_test = x[test_ind]
    y_test = y[test_ind]

    return x_train, y_train, x_test, y_test

def forward_model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(n_feat, input_dim=n_feat, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(n_classes, kernel_initializer='normal', activation='softmax'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_scores(scores, save=False):
    f = plt.figure(1)
    plt.plot(scores['acc'])
    plt.plot(scores['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()
    if save:
        f.savefig('scores.png')

def plot_weights_biases(model, save=False):
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
            wb.append(('B'+str(layer), max_val, 'c'))
            wb.append(('B'+str(layer), min_val, 'y'))
            wb.append(('B'+str(layer), mean_val, 'm'))

        else:
            wb.append(('W'+str(layer), max_val, 'c'))
            wb.append(('W'+str(layer), min_val, 'y'))
            wb.append(('W'+str(layer), mean_val, 'm'))

    labels = [v[0] for v in wb]
    values = [v[1] for v in wb]
    colors = [v[2] for v in wb]
    plt.scatter(labels, values, c=colors)

    plt.title('Analyzing weights and biases')
    plt.ylabel('Values')
    plt.ylim(-5,5)
    elements = \
        [Line2D([0], [0], marker='o', color='c', label='Maximum'),
        Line2D([0], [0], marker='o', color='y', label='Minimum'),
        Line2D([0], [0], marker='o', color='m', label='Mean')]
    plt.legend(handles=elements, loc='lower left')
    plt.show()
    if save:
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
    model = forward_model()
    print('fit model')
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=20, verbose=2)
    
    # not saving to disk for submission
    plot_scores(history.history)
    plot_weights_biases(model)
    # find_errors(model, x_test, y_test)
