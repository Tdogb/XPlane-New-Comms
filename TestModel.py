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
        self.l0 = nn.Linear(12,200)
        self.l1 = nn.Linear(200,200)
        self.l2 = nn.Linear(200,6)

    def forward(self, x):
        A = self.relu0(self.l0(x))
        B = self.relu0(self.l1(A))
        C = self.l2(B)
        return C

print("HELLO")
net = Net()#.cuda()
net.load_state_dict(torch.load('D:\Documents\VScode\Core Lab\XPlane Sim\\nn_output-LOSS2'))
print('D:\Documents\VScode\Core Lab\XPlane Sim\\nn_output-LOSS2')
# net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

inputs = []
outputs = []
def readCSVs():
    with open('D:\Documents\VScode\Core Lab\XPlane Sim\XPlane-Inputs-2.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            inputs.append(list(map(float,row)))
    with open('D:\Documents\VScode\Core Lab\XPlane Sim\XPlane-Outputs-2.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # ignoreIdxs = [0,1,8,9]
        ignoreIdxs = [0,0,1,1,1,1,1,1,0,0]
        for row in reader:
            rw = itertools.compress(row, ignoreIdxs)
            outputs.append(list(map(float,rw)))

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
        print("i:", idx, "diff:", y_test_i.data.numpy()-y_pred.data.numpy())
        print(loss.item())
    return losses_test
print("testing")
readCSVs()
test(inputs, outputs)