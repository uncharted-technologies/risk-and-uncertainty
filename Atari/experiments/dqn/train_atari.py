#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains DQN on Breakout for 200M frames

"""
from agents.common.networks import CNNDeepmind
from agents.dqn.dqn import DQN
from agents.atari_wrappers import make_atari, wrap_deepmind

import pickle
import numpy as np
import matplotlib.pyplot as plt

notes = "This is a test run"

env = make_atari("BreakoutNoFrameskip-v4",noop=False)
env = wrap_deepmind(env, episode_life=False)

nb_steps = 50000000

agent = DQN( env,
                 CNNDeepmind,
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
                 logging=True,
                 log_folder_details="Breakout-DQN",
                 notes=notes)


agent.learn(timesteps=nb_steps, verbose=True)
