
import pickle
import torch
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl_baselines.baselines import EQRDQN
from rl_baselines.common.networks import CNNDeepmind_Multihead

from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind


AGENT_PATH = '???'

env = make_atari("BreakoutNoFrameskip-v4",noop=False)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = EQRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)

agent.load(AGENT_PATH)


obs = env.reset()
returns = 0
for i in range(10000):
    net1,net2 = agent.network(torch.FloatTensor(obs))
    net1 = net1.view(agent.env.action_space.n,agent.n_quantiles)
    net2 = net2.view(agent.env.action_space.n,agent.n_quantiles)
    
    uncertainty = torch.sqrt(torch.mean((net1-net2)**2,dim=1)/2).detach()
    means = torch.mean((net1+net2)/2,dim=1).detach()
    if np.random.uniform() < 0.001:
        action = np.random.choice(agent.env.action_space.n)
    else:
        action = agent.predict(torch.FloatTensor(obs))
    obs, rew, done, info = env.step(action)
    env.render()
    returns += rew
    if done:
        obs = env.reset()
        print(returns)
        returns = 0

