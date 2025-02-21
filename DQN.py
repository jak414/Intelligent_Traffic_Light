"""DQN"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, concatenate
import numpy as np
import random
from collections import deque

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class DQNetwork_flat(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name, chkpt_dir="results"):
        super(DQNetwork_flat, self).__init__()
   
        self.checkpoint_dir = chkpt_dir[0]
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
    
        self.input_dims = input_dims

        self.fc1 = nn.Linear(self.input_dims, 64) #64 32 3
        self.fc2 = nn.Linear(64, 16)
        #self.fc3 = nn.Linear(32, 16)
        #self.fc4 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)  #Adam
        self.loss = nn.MSELoss() 
        #if n_actions>2:
         #   self.loss = nn.CrossEntropyLoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        layer1 = F.leaky_relu(self.fc1(T.tensor(state).to(T.float32)))
        layer2 = F.leaky_relu(self.fc2(layer1))
        #layer3 = F.relu(self.fc3(layer2))
        #layer4 = F.relu(self.fc4(layer3))
        out = self.fc3(layer2)

        return out
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=T.device('cpu')))


class DQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name, chkpt_dir="results"):
        super(DQNetwork, self).__init__()
        #/home/giacomo/tesi/tesi_giacomo/results
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.conv1 = nn.Conv2d(input_dims[0], 4, 2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, 1, stride=2)
        self.conv3 = nn.Conv2d(8, 16, 1, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc4 = nn.Linear(fc_input_dims+3, 64)
        self.fc5 = nn.Linear(64, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state_m, state_s):
        layer1 = F.relu(self.conv1(state_m.to(dtype=T.float32)))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))
        # layer3 shape is BS x n_filters x H x W
        # first dim is batch size, then -1 means that we flatten the other dims
        layer4 = layer3.view(layer3.size()[0], -1) 
        layer4 = F.relu(self.fc4(T.cat((layer4, state_s), dim=1)))

        out = self.fc5(layer4)

        return out

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=T.device('cpu')))
