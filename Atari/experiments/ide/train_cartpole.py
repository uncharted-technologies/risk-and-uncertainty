#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents.ide.ide import IDE
from agents.common.networks.mlp import MLP_Multihead

notes = "This is a test run."

env = gym.make("CartPole-v0")

nb_steps = 5000

agent = IDE( env,
                 MLP_Multihead,
                 n_quantiles=20,
                 kappa=10,
                 prior=0.01,
                 lamda=0.2,
                 replay_start_size=50,
                 replay_buffer_size=50000,
                 gamma=0.99,
                 update_target_frequency=50,
                 minibatch_size=32,
                 learning_rate=5e-3,
                 adam_epsilon=1e-8,
                 update_frequency=1,
                 logging=True,
                 log_folder_details="Cartpole-IDE",
                 notes=notes)


agent.learn(timesteps=nb_steps, verbose=True)

scores = np.array(agent.logger.memory['Episode_score'])
plt.cla()
plt.plot(scores[:,1],scores[:,0])
plt.show()