import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
import sys

seed = 7
np.random.seed(seed)

n_epoch = 3000 
n_class = 2
batch_size = 100
learning_rate = 1e-3

model_path = 'model_adult.pt'
scores_path = 'scores_adult.npy'

# CUDA for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomData(Dataset):
    def __init__(self, images, labels, ids):
        self.labels = labels
        self.images = images
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.tensor(self.images[index]).float()
        label = self.labels[index]
        i = self.ids[index]
        return image, label, i

def load_data(split=0.15):
    x = np.load('./Adult/data.npy')
    y = np.load('./Adult/labels.npy')
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    size = x.shape[0]
    test = int(split*size)
    ind = np.random.permutation(size)
    train_ind = ind[:-test]
    test_ind = ind[-test:]
    x_train = x[train_ind]
    y_train = y[train_ind]
    x_test = x[test_ind]
    y_test = y[test_ind]

    train = CustomData(x_train, y_train, train_ind)
    test = CustomData(x_test, y_test, test_ind)

    # print(train.images.shape)
    # print(train.labels.shape)
    train_loader = DataLoader(dataset=train, 
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=train, 
        batch_size=batch_size, shuffle=False)


    return train_loader, test_loader

class NeuralNet(nn.Module):
    def __init__(self, n_class):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(67, 100)
        self.fc2 = nn.Linear(100, 500)
        self.fc3 = nn.Linear(500, n_class)
        self.nn = nn.Sequential(
            self.fc1, nn.ReLU(), 
            self.fc2, nn.ReLU(),
            self.fc3, nn.Tanh(), 
        )

    def forward(self, x):
        out = self.nn(x)
        return out

class NeuralNet2(nn.Module):
    def __init__(self, n_class):
        super(NeuralNet2, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=67)
        self.fc1 = nn.Linear(67, 100)
        self.fc2 = nn.Linear(100, 500) 
        self.fc3 = nn.Linear(500, 1000)
        self.fc4 = nn.Linear(1000, 500)
        self.fc5 = nn.Linear(500, 1000)
        self.fc6 = nn.Linear(1000, 500)
        self.fc7 = nn.Linear(500, 100)
        self.fc8 = nn.Linear(100, n_class)
        self.nn = nn.Sequential(
            nn.BatchNorm1d(num_features=67),
            self.fc1, nn.ReLU(), 
            self.fc2, nn.ReLU(), 
            self.fc3, nn.ReLU(), nn.Dropout(0.3),
            self.fc4, nn.ReLU(),
            self.fc5, nn.ReLU(), nn.Dropout(0.3),
            self.fc6, nn.ReLU(),
            self.fc7, nn.ReLU(),
            self.fc8, nn.Tanh()
        )

    def forward(self, x):
        out = self.bn(x)
        out = self.nn(out)
        return out


def train():
    model.train()
    correct = 0
    total = 0
    for i, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        labels = labels.long()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct/total
    return train_acc


def find_errors(predicted, labels, ids, limit=10):
    with open('Adult/errors.txt', 'w') as f:
        errors = (predicted != labels).nonzero()
        errors = errors[: min(limit, len(errors))]
        for error in errors:
            e = error.item()
            print('Image {} should have label {} but predicted as {}'\
                    .format(ids[e], labels[e], predicted[e]), file=f)



def test(errors=False):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, ids in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if errors:
                find_errors(predicted, labels, ids)
        test_acc = correct/total
        return test_acc


def plot_scores(scores, save=True):
    f = plt.figure(1)
    train_acc = [i[0] for i in scores]
    test_acc = [i[1] for i in scores]
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    if save:
        f.savefig('Adult/scores.png')

def plot_params(model, save=True):
    f = plt.figure(2)
    wb = []
    for i, p in enumerate(model.parameters()):
        max_val = p.max().item()
        min_val = p.min().item()
        mean_val = p.mean().item()
        wb.append((i, max_val, 'c'))
        wb.append((i, min_val, 'y'))
        wb.append((i, mean_val, 'm'))

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
    plt.ylim(-3,3)
    plt.show()
    if save:
        f.savefig('Adult/wb.png')


if __name__ == '__main__':
    load = int(sys.argv[1])
    model = NeuralNet(n_class).to(device)

    if load:
        print('loading')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        scores = np.load(scores_path)

    else:
        train_loader, test_loader = load_data()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        scores = []
        for epoch in range(n_epoch):
            train_acc = train()
            test_acc = test()
            scores.append([train_acc, test_acc])
            if (epoch+1) % 5 == 0 or epoch == n_epoch-1:
                print('Epoch {} train_acc: {}, test_acc: {}'.
                    format(epoch+1, train_acc, test_acc))

        np.save(scores_path, np.array(scores))
      
        torch.save(model.state_dict(), model_path)

    plot_scores(scores)
    plot_params(model)
    test(errors=True)








