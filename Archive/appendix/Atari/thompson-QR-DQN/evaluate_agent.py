#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import torch
import time
import random

import gym
import matplotlib.pyplot as plt
import numpy as np

from qrdqn import QRDQN
from cnn_deepmind import CNNDeepmind_Multihead

from atari_wrappers import make_atari, wrap_deepmind

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game_scores = []

env = make_atari("BreakoutNoFrameskip-v4",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = QRDQN(env,CNNDeepmind,n_quantiles=200)

for i in range(50):

    filename = 'network_' + str((i+1)*250000) + '.pth'

    agent.load(filename)

    score = 0
    scores = []
    timeout_reached = False
    timestep = 0
    while not timeout_reached:
        this_episode_time = 0
        done = False
        obs = env.reset()
        lives = env.unwrapped.ale.lives()
        while not done:
            if env.unwrapped.ale.lives() < lives:
                lives = env.unwrapped.ale.lives()
                if env.unwrapped.get_action_meanings()[1] == 'FIRE':
                    obs, rew, done, info = env.step(1)
            else:
                if np.random.uniform() < 0.001:
                    action = np.random.choice(agent.env.action_space.n)
                else:
                    action = agent.predict(torch.FloatTensor(obs).to(device))
                obs, rew, done, info = env.step(action)
            score += rew

            if done:
                scores.append(score)
                score = 0

            timestep += 1
            this_episode_time += 1

            if this_episode_time == 27000:
                done = True

            if timestep == 125000:
                timeout_reached = True
                done = True

    print(np.mean(scores))
    game_scores.append(np.mean(scores))

with open('scores', 'wb') as output:
    pickle.dump(game_scores, output, pickle.HIGHEST_PROTOCOL)