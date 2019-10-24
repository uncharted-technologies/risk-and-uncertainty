"""
Modified from : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random
from collections import namedtuple

import torch


Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'state_next', 'done')
)


class ReplayBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = {}
        self._position = 0

    def add(self, *args):
        """
        Saves a transition.
        """
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, device=None):
        """
        Samples a batch of Transitions, with the tensors already stacked
        and transfered to the specified device.
        Return a list of tensors in the order specified in Transition.
        """
        batch = random.sample(list(self._memory.values()), batch_size)
        return [torch.stack(tensors).to(device) for tensors in zip(*batch)]

    def __len__(self):
        return len(self._memory)
