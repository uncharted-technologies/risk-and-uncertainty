#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
import pickle
import torch
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents.qrdqn.qrdqn import QRDQN
from agents.common.networks.mlp import MLP

AGENT_PATH = '???'

env = gym.make("CartPole-v0")

agent = QRDQN(env,MLP,n_quantiles=20)

agent.load(AGENT_PATH)

obs = env.reset()
returns = 0
for i in range(10000):
    out = agent.network(torch.FloatTensor(obs))
    out = out.view(agent.env.action_space.n,agent.n_quantiles)
    out = np.array(out.detach())
    plt.cla()
    plt.plot(out[0,:], label="right")
    plt.plot(out[1,:], label="left")
    plt.legend()
    plt.draw()
    plt.pause(0.01) 
    action = agent.predict(torch.FloatTensor(obs))
    obs, rew, done, info = env.step(action)
    env.render()
    time.sleep(0.2)
    returns += rew
    if done:
        obs = env.reset()
        print(returns)
        returns = 0