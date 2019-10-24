
import torch
import time

from rl_baselines.common.networks import CNNDeepmind
from rl_baselines.baselines import QRDQN
from rl_baselines.envs.atari_wrappers import make_atari, wrap_deepmind

import pickle
import numpy as np
import matplotlib.pyplot as plt

AGENT_PATH = '???'

env = make_atari("BreakoutNoFrameskip-v4",noop=True)
env = wrap_deepmind(env,episode_life=True)

agent = QRDQN(env, CNNDeepmind, n_quantiles=200)

agent.load(AGENT_PATH)

obs = env.reset()
score = 0
scores = []
lives = env.unwrapped.ale.lives()
for i in range(1000000):
    if env.unwrapped.ale.lives() < lives:
        lives = env.unwrapped.ale.lives()
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            obs, rew, done, info = env.step(1)
    else:
        if np.random.uniform() < 0.01:
            action = np.random.choice(agent.env.action_space.n)
        else:
            action = agent.predict(torch.FloatTensor(obs))
        obs, rew, done, info = env.step(action)
    score += rew
    env.render()
    #time.sleep(0.02)
    if done:
        obs = env.reset()
        lives = env.unwrapped.ale.lives()
        print(score)
        scores.append(score)
        score=0

print(np.mean(scores))