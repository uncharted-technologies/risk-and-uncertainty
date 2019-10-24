#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains DQN on Cartpole
"""

import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents.dqn.dqn import DQN
from agents.common.networks.mlp import MLP

notes = "This is a test run"

env = gym.make("CartPole-v0")

nb_steps = 5000

agent = DQN( env,
                 MLP,
                 replay_start_size=50,
                 replay_buffer_size=50000,
                 gamma=0.99,
                 update_target_frequency=50,
                 minibatch_size=32,
                 learning_rate=1e-3,
                 initial_exploration_rate=1,
                 final_exploration_rate=0.02,
                 final_exploration_step=1000,
                 adam_epsilon=1e-8,
                 update_frequency=1,
                 logging=True,
                 log_folder_details="Cartpole-DQN",
                 loss='mse',
                 notes=notes)


agent.learn(timesteps=nb_steps, verbose=True)

scores = np.array(agent.logger.memory['Episode_score'])
plt.cla()
plt.plot(scores[:,1],scores[:,0])
plt.show()
