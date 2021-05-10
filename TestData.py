import enum
from torch._C import dtype
from torch.utils.data.dataloader import DataLoader
import xpc
import sys, threading, time, math, random
from numpy.lib.function_base import append
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.tensor import Tensor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.relu0 = nn.ReLU()
        self.l0 = nn.Linear(12,128)
        self.l1 = nn.Linear(128,128)
        self.l2 = nn.Linear(128,10)

    def forward(self, x):
        A = self.relu0(self.l0(x))
        B = self.relu0(self.l1(A))
        C = self.l2(B)
        return C
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
# net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

inputs = []
outputs = []

def readCSVs():
    with open('D:\Documents\VScode\XPlane-Inputs.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            inputs.append(list(map(float,row)))
    with open('D:\Documents\VScode\XPlane-Outputs.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            outputs.append(list(map(float,row)))
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)

def train():
    net.train()
    losses = []
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(inputs, outputs, train_size=0.9)
    print(x_train_t[0])
    dataset = TensorDataset(torch.FloatTensor(x_train_t), torch.FloatTensor(y_train_t))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(1,30000):
        for idx, (x,y) in enumerate(dataloader):
            # x = x.to(device)
            # y = y.to(device)
            x_train = Variable(x).float()
            y_train = Variable(y).float()
            optimizer.zero_grad()
            y_pred = net(x_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % 10:
            print(np.mean(losses[-10:]))
    torch.save(net.state_dict(), "nn_output")
    return losses, x_test_t, y_test_t

def nn():
    # print(states[59])
    # print("controls: ", controls[59])
    losses, x_test, y_test = train()
    fig, axs = plt.subplots(3)
    axs[0].plot(range(0, len(losses)), losses)
    # axs[0].set_title("Training loss")
    new = losses[500:len(losses)-1]
    axs[1].plot(range(0, len(new)), new)
    # axs[1].set_title("zoomed in training loss")
    losses_test = test(x_test, y_test)
    axs[2].plot(range(0, len(losses_test)), losses_test)
    # axs[2].set_title("Testing Loss")
    plt.show()

def test(x_test, y_test):
    print("----------TESTING-----------")
    losses_test = []
    dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, (x,y) in enumerate(dataset):
        x_test_i = Variable(x).float()
        y_test_i = Variable(y).float()
        y_pred = net(x_test_i)
        loss = criterion(y_test_i, y_pred)
        losses_test.append(loss.item())
        print("i:", idx, " real: ", y_test_i, " pred: ", y_pred)
        print(loss.item())
    return losses_test

readCSVs()
train()
nn()