import numpy as np
# import matplotlib
# matplotlib.use('agg') 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

seed = 7
np.random.seed(seed)

n_epoch = 20
n_class = 5
batch_size = 100
learning_rate = 1e-3

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
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return image, label

def load_data(split=0.15):
    x = np.load('./Flowers/flower_imgs.npy')
    x = x/255.0
    y = np.load('./Flowers/flower_labels.npy')
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



    train = CustomData(x_train, y_train, train_ind, transforms)
    test = CustomData(x_test, y_test, test_ind, transforms)

    print(train.images)
    print(train.labels)
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

        self.pool2 = nn.MaxPool2d(kernel_size=2)


        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2)

        self.fc = nn.Linear(in_features=64, out_features=n_class)
        self.sm = nn.Softmax()

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 64)
        output = self.fc(output)
        output = self.sm(output)
        return output



train_loader, test_loader = load_data()
model = CNNet(n_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def train():
    model.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct/total
    return train_acc


def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = correct/total
        return test_acc



if __name__ == '__main__':

    history = []
    for epoch in range(n_epoch):
        train_acc = train()
        test_acc = test()
        history.append([train_acc, test_acc])
        if (epoch+1) % 1 == 0 or epoch == n_epoch-1:
            print('epoch {}: , train_acc: {}, test_acc: {}'.
                format(epoch, train_acc, test_acc))

    np.save('history.npy', np.array(history))


  
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')


