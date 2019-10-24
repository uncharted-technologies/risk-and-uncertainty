
import pickle
import torch
import time
import random

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents.eqrdqn.eqrdqn import EQRDQN
from agents.common.networks.cnn_deepmind import CNNDeepmind_Multihead

from agents.common.atari_wrappers import make_atari, wrap_deepmind

FOLDER = "???"

game_scores = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_atari("BreakoutNoFrameskip",noop=True)
env = wrap_deepmind(env,clip_rewards=False,episode_life=False)

agent = EQRDQN(env,CNNDeepmind_Multihead,n_quantiles=200)


for i in range(50):

    filename = "network_" + str((i+1)*250000) + ".pth"

    agent.load(FOLDER+filename)

    score = 0
    scores = []
    total_timesteps = 0
    while total_timesteps < 125000:
        done = False
        obs = env.reset()
        lives = env.unwrapped.ale.lives()
        this_episode_time = 0
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
            this_episode_time += 1
            total_timesteps += 1

            if this_episode_time == 27000:
                done = True

            if done:
                #print(score)
                scores.append(score)
                score = 0
                i += 1

    print(np.mean(scores))
    game_scores.append(np.mean(scores))

    pickle.dump(game_scores, open(FOLDER+"test_scores", "wb" ) )