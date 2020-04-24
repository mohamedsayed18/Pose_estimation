import numpy as np
from numpy import loadtxt
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt

from helpers import *


#model = tf.lite.Interpreter("posenet.tflite")
#load_image('Data_set/Prayer_pose/img0.jpg')
#get_vectors('Data_set/Prayer_pose', model)

# Build new Network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.l1 = nn.Linear(34, 20)
        self.l2 = nn.Linear(20, 10)
        self.l3 = nn.Linear(10, 3)
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.sig(self.l1(x))
        x = self.sig(self.l2(x))
        x = self.sig(self.l3(x))
        return x

def train(net, data_set):
    criterion = nn.CrossEntropyLoss()   # cross entropy
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # optimizer
    vector, label = data_set[:, :-1], data_set[:, -1]
    for i in range(data_set.shape[0]):
        inputs, labels = torch.Tensor(vector[i].reshape(1,34)), torch.tensor([np.long(label[i])])
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return

mynet = MyNetwork()
# Load the data for the classifier
data = loadtxt('poses.csv', delimiter=',', dtype=np.float)
#inputs, labels = torch.Tensor(data[:,:-1]), torch.Tensor([data[:,-1]])
train(mynet, data)

