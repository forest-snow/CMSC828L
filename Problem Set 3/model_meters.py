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

def load_data():
    x = np.loadtxt('./Three Meter/data.csv', delimiter=',')
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    size = x.shape[0]
    ind = np.random.permutation(size)
    x_train = x[ind]

    train = CustomData(x_train, ind)

    train_loader = DataLoader(dataset=train, 
        batch_size=batch_size, shuffle=False)

    return train_loader

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(33, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 33),
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


# def find_errors(predicted, labels, ids, limit=10):
#     with open('Three Meter/errors.txt', 'w') as f:
#         errors = (predicted != labels).nonzero()
#         errors = errors[: min(limit, len(errors))]
#         for error in errors:
#             e = error.item()
#             print('Image {} should have label {} but predicted as {}'\
#                     .format(ids[e], labels[e], predicted[e]), file=f)



# def test(errors=False):
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, ids in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             if errors:
#                 find_errors(predicted, labels, ids)
#         test_acc = correct/total
#         return test_acc


def plot_scores(scores, save=True):
    f = plt.figure(1)
    train_acc = [i for i in scores]
    plt.plot(train_acc)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['MAE'], loc='upper right')
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
    load = int(sys.argv[1])
    model = AutoEncoder().to(device)

    if load:
        print('loading')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        scores = np.load(scores_path)

    else:
        train_loader = load_data()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss()

        scores = []
        for epoch in range(n_epoch):
            train_acc = train()
            scores.append(train_acc)
            if (epoch+1) % 5 == 0 or epoch == n_epoch-1:
                print('Epoch {} train_acc: {}'.
                    format(epoch+1, train_acc))

        np.save(scores_path, np.array(scores))

        torch.save(model.state_dict(), model_path)

    plot_scores(scores)
    plot_params(model)
    # test(errors=True)








