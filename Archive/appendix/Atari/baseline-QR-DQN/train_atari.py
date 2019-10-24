#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run this script to train a QR-DQN agent in the selected environment

"""
from cnn_deepmind import CNNDeepmind
from qrdqn import QRDQN
from atari_wrappers import make_atari, wrap_deepmind

import pickle
import numpy as np
import matplotlib.pyplot as plt

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env)

nb_steps = 12500000

agent = QRDQN( env,
                 CNNDeepmind,
                 n_quantiles=200,
                 kappa=0,
                 replay_start_size=50000,
                 replay_buffer_size=1000000,
                 gamma=0.99,
                 update_target_frequency=10000,
                 minibatch_size=32,
                 learning_rate=5e-5,
                 initial_exploration_rate=1,
                 final_exploration_rate=0.01,
                 final_exploration_step=1000000,
                 adam_epsilon=0.01/32,
                 update_frequency=4,
                 logging=True)


agent.learn(timesteps=nb_steps, verbose=True)
agent.save()


