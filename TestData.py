import enum
from torch._C import dtype
from torch.utils.data.dataloader import DataLoader
import xpc
import sys, threading, time, math, random, itertools
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
        self.l0 = nn.Linear(10,128)
        self.l1 = nn.Linear(128,64)
        self.l2 = nn.Linear(64,6)

    def forward(self, x):
        A = self.relu0(self.l0(x))
        B = self.relu0(self.l1(A))
        C = self.l2(B)
        return C

class Nominal(nn.Module):
    def __init__(self, input_size=12, hidden_layer_size=200, state_size=7):
        super().__init__()
        torch.manual_seed(0)
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.linearStates = nn.Linear(hidden_layer_size, state_size)

    def forward(self, input_seq):
        self.hidden_cell = (torch.zeros(1, input_seq.shape[0], self.hidden_layer_size),
                            torch.zeros(1, input_seq.shape[0], self.hidden_layer_size))
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictionStates = self.linearStates(lstm_out)

        return predictionStates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = Net()#.cuda()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

inputs = []
outputs = []

def readCSVs():
    with open('D:\Documents\VScode\Core Lab\XPlane Sim\XPlane-Inputs-2.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        ignoreIdxs = [1,1,1,1,1,1,1,1,1,0,0,1]
        for row in reader:
            rw = itertools.compress(row, ignoreIdxs)
            inputs.append(list(map(float,row)))
    with open('D:\Documents\VScode\Core Lab\XPlane Sim\XPlane-Outputs-2.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # ignoreIdxs = [0,1,8,9]
        ignoreIdxs = [1,1,1,1,0,0,0,1,1,0] # ignoreIdxs = [0,0,1,1,1,0,0,1,0,0]
        for row in reader:
            rw = itertools.compress(row, ignoreIdxs)
            outputs.append(list(map(float,rw)))
            
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)

def train():
    net.train()
    losses = []
    validation_losses = []
    validationZeroCounter = 0
    validation_loss = 0
    previousValidationLoss = 999999
    validation_diff = 0
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(inputs, outputs, train_size=0.8)
    dataset = TensorDataset(torch.FloatTensor(x_train_t), torch.FloatTensor(y_train_t))
    batch_size = 50
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_validation = TensorDataset(torch.FloatTensor(x_test_t), torch.FloatTensor(y_test_t))
    dataloader_validation = DataLoader(dataset_validation, batch_size=1, shuffle=True)
    for epoch in range(1,37500):
        for idx, (x,y) in enumerate(dataloader):
            # x = x.to(device)
            # y = y.to(device)
            x_train = Variable(x).float().to(device)
            y_train = Variable(y).float().to(device)
            optimizer.zero_grad()
            # net.hidden_cell = (torch.zeros(1, batch_size, net.hidden_layer_size),
            #                      torch.zeros(1, batch_size, net.hidden_layer_size),)
            y_pred = net(x_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            validation_losses = []
            for idx, (x,y) in enumerate(dataset_validation):
                x_test_i = Variable(x).float().to(device)
                y_test_i = Variable(y).float().to(device)
                y_pred = net(x_test_i)
                loss = criterion(y_test_i, y_pred)
                validation_losses.append(loss.item())
            validation_loss = np.mean(validation_losses)
            if previousValidationLoss - validation_loss == 0:
                print(validationZeroCounter)
                print(type(validationZeroCounter))
                validationZeroCounter += 1
                if validationZeroCounter == 50:
                    torch.save(net.state_dict(), "D:\Documents\VScode\Core Lab\XPlane Sim\\nn_output")
                    input()
                    break
            else:
                validationZeroCounter = 0
            validation_diff = previousValidationLoss - validation_loss
            previousValidationLoss = validation_loss
        print("Epoch: ", epoch, "Loss", np.mean(losses[-160:]), "Validation loss change", validation_diff)
    torch.save(net.state_dict(), "D:\Documents\VScode\Core Lab\XPlane Sim\\nn_output")
    return losses, x_test_t, y_test_t

def nn():
    # print(states[59])
    # print("controls: ", controls[59])
    losses, x_test, y_test = train()
    fig, axs = plt.subplots(3)
    axs[0].plot(range(0, len(losses)), losses)
    new = losses[500:len(losses)-1]
    axs[1].plot(range(0, len(new)), new)
    # losses_test = test(x_test, y_test)
    # axs[2].plot(range(0, len(losses_test)), losses_test)
    plt.show()

def test(x_test, y_test):
    print("----------TESTING-----------")
    losses_test = []
    dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, (x,y) in enumerate(dataset):
        x_test_i = Variable(x).float().to(device)
        y_test_i = Variable(y).float().to(device)
        y_pred = net(x_test_i)
        loss = criterion(y_test_i, y_pred)
        losses_test.append(loss.item())
        print("i:", idx, " real: ", y_test_i, " pred: ", y_pred)
        print(loss.item())
    return losses_test
if __name__ == "__main__":
    readCSVs()
    nn()