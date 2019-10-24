import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)

class CNNDeepmind(nn.Module):

    def __init__(self, observation_space, n_outputs, width=512,weight_scale=np.sqrt(2),**kwargs):
        # CNN architechture of DeepMind described in https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf :

        # The first hidden layer convolves 32 filters of 8 3 8 with stride 4 with the input image and applies a rectifier nonlinearity
        # The second hidden layer convolves 64 filters of 4 3 4 with stride 2, again followed by a rectifier nonlinearity
        # This is followed by a third convolutional layer that convolves 64 filters of 3 3 3 with stride 1 followed by a rectifier.
        # The final hidden layer is fully-connected and consists of 512 rectifier units.

        super().__init__()

        self.weight_scale = weight_scale

        if len(observation_space.shape) != 3:
            raise NotImplementedError

        # Defining the network architechture
        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.output = nn.Sequential(nn.Linear(64 * 7 * 7, width),
                                  nn.ReLU(),
                                  nn.Linear(width, n_outputs))

        self.conv.apply(lambda x: init_weights(x, self.weight_scale))
        self.output.apply(lambda x: init_weights(x, self.weight_scale))

    def forward(self, obs):
        if len(obs.shape) != 4:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0,3,1,2)
        obs = obs/255
        obs = self.conv(obs)
        obs = obs.view(obs.size(0), -1)
        return self.output(obs)

class CNNDeepmind_Multihead(nn.Module):

    def __init__(self, observation_space, n_outputs_1, n_outputs_2, width=512,weight_scale=np.sqrt(2),):
        # CNN architechture of DeepMind described in https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf :

        # The first hidden layer convolves 32 filters of 8 3 8 with stride 4 with the input image and applies a rectifier nonlinearity
        # The second hidden layer convolves 64 filters of 4 3 4 with stride 2, again followed by a rectifier nonlinearity
        # This is followed by a third convolutional layer that convolves 64 filters of 3 3 3 with stride 1 followed by a rectifier.
        # The final hidden layer is fully-connected and consists of 512 rectifier units.

        super().__init__()

        self.weight_scale = weight_scale

        if len(observation_space.shape) != 3:
            raise NotImplementedError

        # Defining the network architechture
        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.output_1 = nn.Sequential(nn.Linear(64 * 7 * 7, width),
                            nn.ReLU(),
                            nn.Linear(width, n_outputs_1))
        
        self.output_2 = nn.Sequential(nn.Linear(64 * 7 * 7, width),
                            nn.ReLU(),
                            nn.Linear(width, n_outputs_2))

        self.conv.apply(lambda x: init_weights(x, self.weight_scale))

        self.output_1.apply(lambda x: init_weights(x, self.weight_scale))
        self.output_2.apply(lambda x: init_weights(x, self.weight_scale))

    def forward(self, obs):
        if len(obs.shape) != 4:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0,3,1,2)
        obs = obs/255
        obs = self.conv(obs)
        obs = obs.view(obs.size(0), -1) # Flatten the conv output

        return self.output_1(obs), self.output_2(obs)
<