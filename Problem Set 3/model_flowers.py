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

n_epoch = 100
n_class = 5
batch_size = 100
learning_rate = 1e-3

model_path = 'model_flower.pt'
scores_path = 'scores_flower.npy'

# CUDA for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

class CustomData(Dataset):
    def __init__(self, images, labels, ids, transforms):
        self.labels = labels
        self.images = images
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        i = self.ids[index]
        if self.transforms is not None:
            image = self.transforms(image)
            # label = self.transforms(label)

        return image, label, i

def load_data(split=0.15):
    x = np.load('./Flowers/flower_imgs.npy')
    # x = x/255.0
    y = np.load('./Flowers/flower_labels.npy')
    # y = to_categorical(y)

    size = x.shape[0]
    test = int(split*size)
    ind = np.random.permutation(size)
    train_ind = ind[:-test]
    test_ind = ind[-test:]
    x_train = x[train_ind]
    y_train = y[train_ind]
    x_test = x[test_ind]
    y_test = y[test_ind]



    train = CustomData(x_train, y_train, train_ind, transforms)
    test = CustomData(x_test, y_test, test_ind, transforms)

    # print(train.images)
    # print(train.labels)
    train_loader = DataLoader(dataset=train, 
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=train, 
        batch_size=batch_size, shuffle=False)


    return train_loader, test_loader

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class CNNet(nn.Module):
    def __init__(self, n_class):
        super(CNNet, self).__init__()

        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.AvgPool2d(kernel_size=4)


        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit6
                                 , self.unit7, self.pool2)

        self.fc = nn.Linear(in_features=64, out_features=n_class)
        # self.sm = nn.Softmax()

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 64)
        output = self.fc(output)
        return output






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
    with open('Flowers/errors.txt', 'w') as f:
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
        f.savefig('Flowers/scores.png')

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
        f.savefig('Flowers/wb.png')


if __name__ == '__main__':
    load = int(sys.argv[1])
    model = CNNet(n_class).to(device)

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


      
        # Save the model checkpoint
        torch.save(model.state_dict(), model_path)

    plot_scores(scores)
    plot_params(model)
    test(errors=True)








