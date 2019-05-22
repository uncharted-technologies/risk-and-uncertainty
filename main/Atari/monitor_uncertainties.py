#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads a trained agent ("network.pth"), runs it for an episode, 
and saves both the uncertainties and the game frames. It also both renders 
the environment and plots the uncertainty while the agent is playing.

"""
import pickle
import torch
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from qrdqn import QRDQN
from cnn_deepmind import CNNDeepmind_Multihead

from atari_wrappers import make_atari, wrap_deepmind

env = make_atari("BreakoutNoFrameskip-v4",noop=False)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = QRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)

agent.load('agent.pth')

returns = 0
done = False
obs = env.reset()
lives = env.unwrapped.ale.lives()
this_episode_time = 0
uncertainties = []
means = []
stds=[]
deaths = []
while not done:
    net1,net2 = agent.network(torch.FloatTensor(obs))
    net1 = net1.view(agent.env.action_space.n,agent.n_quantiles)
    net2 = net2.view(agent.env.action_space.n,agent.n_quantiles)
   
    if env.unwrapped.ale.lives() < lives:
        lives = env.unwrapped.ale.lives()
        deaths.append(this_episode_time)
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            obs, rew, done, info = env.step(1)
    else:
        if np.random.uniform() < 0.05: #Epsilon-greedy policy
            action = np.random.choice(agent.env.action_space.n)
        else:
            action = agent.predict(torch.FloatTensor(obs))
        obs, rew, done, info = env.step(action)

    
    uncertainty = torch.sqrt(torch.mean((net1-net2)**2,dim=1)/2).detach()
    mean = torch.mean((net1+net2)/2,dim=1).detach()
    std = torch.std((net1+net2)/2,dim=1).detach()
    plt.cla()
    uncertainties.append(10*uncertainty[action])
    means.append(mean[action])
    stds.append(std[action])
    plt.plot(uncertainties)
    if deaths:
        for i in deaths:
            plt.scatter(i,0,c='r')
    plt.draw()
    plt.pause(0.01) 

    env.render()
    returns += rew
    this_episode_time += 1
    if this_episode_time == 27000:
        done = True
    
    if (this_episode_time + 1) % 2 == 0:
        env.unwrapped.ale.saveScreenPNG('game_frames/'+str(this_episode_time)+'.png')

    if done:
        print(returns)
        time.sleep(5)

to_save = [uncertainties,deaths]

with open('uncertainties', 'wb') as output:
    pickle.dump(to_save, output, pickle.HIGHEST_PROTOCOL)