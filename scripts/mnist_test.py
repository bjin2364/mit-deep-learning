#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import torchvision

cuda0 = torch.device('cuda:0')
torch.manual_seed(1234)
data_path = "~/Downloads/data"



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def train(net, optim, train_loader):
    net.train()
    for image, label in train_loader:
      # put data onto GPU
      img = image.cuda()
      lab = label.cuda()

      # clear gradient
      net.zero_grad()

      # forward through the network
      output = net.forward(img)

      # compute loss and gradient
      loss = F.cross_entropy(output, lab)
      loss.backward()

      # update parameters
      optim.step()

def evaluate(net, val_loader):
    total = 0
    correct = 0
    
    net.eval()  # puts the network in eval mode. this is important when the 
                # network has layers that behaves differently in training and 
                # evaluation time, e.g., dropout and batch norm.
    for image, label in val_loader:
        # put data onto GPU
        img = image.cuda()
        label = label.cuda()
        
        with torch.no_grad():  # gradients are not tracked in this context manager
                               # since we are evaluating, gradients are not needed 
                               # and we can save some time and GPU memory.
              
            # forward through the network, and get the predicted class
            # FIXME!!!  (HINT: use .argmax(dim=-1))
            #   `prediction` should be an integer vector of size equal to the batch size.
            #   Remember that the network outputs logits of the prediction probabilities, 
            #   and that the higher the logits, the higher the probability.
            prediction = net.forward(img).argmax(dim=-1)
            
            total += image.size(0)  # batch size
            correct += (prediction == label).sum().item()  # `.item()` retreives a python number from a 1-element tensor
            
    return correct / total

def download_data():
    mnist_train = torchvision.datasets.MNIST(root=data_path, download=True, train=True, transform=torchvision.transforms.ToTensor())

    num_pixels = mnist_train.data[0].size()[0] * mnist_train.data[0].size()[1] * len(mnist_train)
    
    batch_size = 10000
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=1)

    total = 0
    for batch in train_loader:
        total += batch[0].sum()
        
    avg = total / num_pixels

    sum_square_error = 0
    for batch in train_loader:
        sum_square_error += ((batch[0] - avg).pow(2)).sum()
    std = torch.sqrt(sum_square_error / num_pixels)

    print("Data set mean: {}, std: {}".format(avg, std))

    norm_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(avg,), std=(std,)),  # [0, 1] range => [-1, 1] range
    ])

    mnist_train = torchvision.datasets.MNIST(root=data_path, download=True, train=True, transform=norm_transform)
    mnist_val = torchvision.datasets.MNIST(root=data_path, download=True, train=False, transform=norm_transform)
    
    return mnist_train, mnist_val

def main():
    train_data, val_data = download_data()

    batch_size = 300
    train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=batch_size,
                                           shuffle=True,                   # shuffle training set
                                           num_workers=4,                  # turns on multi-processing loading so training is not blocked by data loading
                                           pin_memory=True)                # pin_memory allows faster transfer from CPU to GPU
    val_loader = torch.utils.data.DataLoader(val_data, 
                                         batch_size=batch_size, 
                                         num_workers=4, 
                                         pin_memory=True)

    num_epochs = 10
    lr = 0.001

    net = MyNet().to(cuda0)
    optim = Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print('Epoch: {}\tValidation Accuracy: {:.4f}%'.format(epoch, evaluate(net, val_loader) * 100))
        train(net, optim, train_loader)

    print('Done! \tValidation Accuracy: {:.4f}%'.format(evaluate(net, val_loader) * 100))

if __name__ == "__main__":
    main()
