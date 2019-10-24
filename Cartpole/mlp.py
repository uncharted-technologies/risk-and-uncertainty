import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m, gain):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain)


class MLP(torch.nn.Module):
    """
    MLP with ReLU activations after each hidden layer, but not on the output layer
    
    n_output can be None, in that case there is no output layer and so the last layer
    is the last hidden layer, with a ReLU
    """

    def __init__(self, observation_space, n_outputs, hiddens=[100, 100]):
        super().__init__()

        if len(observation_space.shape) != 1:
            raise NotImplementedError
        else:
            n_inputs = observation_space.shape[0]

        layers = []
        for hidden in hiddens:
            layers.append(nn.Linear(n_inputs, hidden))
            layers.append(nn.ReLU())
            n_inputs = hidden

        if n_outputs is not None:
            layers.append(nn.Linear(hidden, n_outputs))

        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        return self.layers(obs)


class MLP_Multihead(torch.nn.Module):
    def __init__(self, observation_space, n_outputs_1, n_outputs_2, width=100):

        super().__init__()

        if len(observation_space.shape) != 1:
            raise NotImplementedError
        else:
            n_inputs = observation_space.shape[0]

        self.output_1 = nn.Sequential(
                            torch.nn.Linear(n_inputs, width),
                            nn.ReLU(),
                            nn.Linear(width, width),
                            nn.ReLU(),
                            nn.Linear(width, n_outputs_1))

        self.output_2 = nn.Sequential(
                            torch.nn.Linear(n_inputs, width),
                            nn.ReLU(),
                            nn.Linear(width, width),
                            nn.ReLU(),
                            nn.Linear(width, n_outputs_2))

        self.output_1.apply(lambda x: init_weights(x, 3))
        self.output_2.apply(lambda x: init_weights(x, 3))

    def forward(self, obs):
        return self.output_1(obs), self.output_2(obs)

