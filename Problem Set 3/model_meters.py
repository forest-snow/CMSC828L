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

n_epoch = 200
batch_size = 500
learning_rate = 1e-3

model_path = 'model_meter.pt'
scores_path = 'scores_meter.npy'

# CUDA for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomData(Dataset):
    def __init__(self, data, ids):
        self.data = data
        self.ids = ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index]).float()
        i = self.ids[index]

        return x, i

def load_data(split=0.15):
    x = np.loadtxt('./Three Meter/data.csv', delimiter=',')
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    size = x.shape[0]
    test = int(split*size)
    ind = np.random.permutation(size)
    train_ind = ind[:-test]
    test_ind = ind[-test:]
    x_train = x[train_ind]
    x_test = x[test_ind]

    train = CustomData(x_train, train_ind)
    test = CustomData(x_test, test_ind)

    train_loader = DataLoader(dataset=train, 
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=train, 
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=33),    
            nn.Linear(33, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 16)
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(num_features=16),
            nn.Linear(16, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 33),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train():
    model.train()
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)

        outputs = model(data)
        loss = criterion(outputs, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def find_errors(data, outputs, ids, limit=10):
    loss = torch.sum((data-outputs).abs(), dim=1)
    with open('Three Meter/errors.txt', 'w') as f:
        i = loss.argmax()
        print('Data {} has loss of {}'\
            .format(ids[i], loss[i]), file=f)
        print('Original: {}'.format(data[i]), file=f)
        print('Recovered: {}'.format(outputs[i]), file=f)



def test(errors=False):
    model.eval()
    with torch.no_grad():
        for i, (data, ids) in enumerate(test_loader):
            data = data.to(device)

            outputs = model(data)
            loss = criterion(outputs, data)
            if errors:
                find_errors(data, outputs, ids)
            return loss.item() 


def plot_scores(scores, save=True):
    f = plt.figure(1)
    train_acc = [i[0] for i in scores]
    test_acc = [i[1] for i in scores]
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('Model loss')
    plt.ylabel('L1 Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    if save:
        f.savefig('Three Meter/scores.png')

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
        f.savefig('Three Meter/wb.png')


if __name__ == '__main__':
    save = True
    load = int(sys.argv[1])
    model = AutoEncoder().to(device)

    if load:
        print('loading')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        scores = np.load(scores_path)

    else:
        train_loader, test_loader = load_data()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss()

        scores = []
        for epoch in range(n_epoch):
            train_acc = train()
            test_acc = test()
            scores.append([train_acc, test_acc])
            if (epoch+1) % 5 == 0 or epoch == n_epoch-1:
                print('Epoch {} train_acc: {}, test_acc: {}'.
                    format(epoch+1, train_acc, test_acc))

        if save:
            np.save(scores_path, np.array(scores))
            torch.save(model.state_dict(), model_path)

    plot_scores(scores, save)
    plot_params(model, save)
    test(errors=save)








