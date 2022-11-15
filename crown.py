"""
   crown.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.hid = hid

        # construct the network
        self.input_size = 2
        self.output_size = 1
        self.fc1 = nn.Linear(self.input_size, self.hid)
        self.fc2 = nn.Linear(self.hid, self.hid)
        self.fc3 = nn.Linear(self.hid, self.output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # self.hid1 = None
        # self.hid2 = None

        x = self.tanh(self.fc1(input))
        self.hid1 = x
        x = self.tanh(self.fc2(x))
        self.hid2 = x
        x = self.sigmoid(self.fc3(x))
        return x


class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.hid = hid

        # construct the network
        self.input_size = 2
        self.output_size = 1
        self.fc1 = nn.Linear(self.input_size, self.hid)
        self.fc2 = nn.Linear(self.hid, self.hid)
        self.fc3 = nn.Linear(self.hid, self.hid)
        self.fc4 = nn.Linear(self.hid, self.output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        # self.hid1 = None
        # self.hid2 = None
        # self.hid3 = None
        x = self.tanh(self.fc1(input))
        self.hid1 = x
        x = self.tanh(self.fc2(x))
        self.hid2 = x
        x = self.tanh(self.fc3(x))
        self.hid3 = x
        x = self.sigmoid(self.fc4(x))
        return x

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()

        self.hid = num_hid

        # construct the network
        self.input_size = 2
        self.output_size = 1
        self.fc1 = nn.Linear(self.input_size, self.hid)
        self.fc2_0 = nn.Linear(self.hid, self.hid)
        self.fc2_1 = nn.Linear(self.hid, self.hid)
        self.fc3_0 = nn.Linear(self.hid, self.output_size)
        self.fc3_1 = nn.Linear(self.hid, self.output_size)
        self.fc3_2 = nn.Linear(self.hid, self.output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # self.hid1 = None
        # self.hid2 = None

        x = self.tanh(self.fc1(input))
        self.hid1 = x
        x = self.tanh(self.fc2_0(x) + self.fc2_1(x))
        self.hid2 = x
        x = self.sigmoid(self.fc3_0(x) + self.fc3_1(x) + self.fc3_2(x))

        return x
