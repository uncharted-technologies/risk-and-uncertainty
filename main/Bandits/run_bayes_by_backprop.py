import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle
import torch
import torch.optim as optim
import torch.autograd as autograd 
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

from function.bayes_by_backprop_functions import *
from eqrdqn_functions import EnvMushroom, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda: ", torch.cuda.is_available())

d = defaultdict(LabelEncoder)
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
data_reduced = pd.read_csv("data/data_mushrooms.csv", index_col=0)

SAMPLES = 2 # Number of samples drawn from the posterior
BATCH_SIZE = 32
n_features = data_reduced.shape[1] - 1
NUM_FRAMES = 20000
l2_reg = 5.
sigma=1.

regret_all_seeds = []
reward_all_seeds = []       

for seed in range(10):
    print('Seed {}'.format(seed))
    env = EnvMushroom(data_reduced)
    replay_buffer = ReplayBuffer(10000)
    net = BayesianNetwork(n_features, device, l2_reg=l2_reg, sigma=sigma)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    regret_cumul = 0
    actions = []
    regrets = []
    reward_cumul = 0
    rewards = []
    n_samples = 0
    for frame_idx in tqdm(range(1, NUM_FRAMES + 1)):
        X = env.sample()
        n_samples += 1
        action, incertitude = act(net, X, SAMPLES)
        actions.append(action)
        reward = env.hit(action)
        regret_cumul += env.regret(action)
        regrets.append(regret_cumul)

        reward_cumul += reward
        rewards.append(reward_cumul)
        replay_buffer.push(X, action, reward)
        
        if frame_idx % 10 == 0:
            if len(replay_buffer.buffer) > 2:
                compute_loss(net, optimizer, replay_buffer, n_samples, SAMPLES, BATCH_SIZE)

    regret_all_seeds.append(regrets)
    reward_all_seeds.append(rewards)
    pickle.dump(regret_all_seeds, open('data/regrets_bayes.p', 'wb'))
    print('Dump regrets : Done')
    pickle.dump(reward_all_seeds, (open('data/rewards_bayes.p', 'wb')))


# for l2_reg in [0.5, 1, 5, 10]:
#     print("l2_reg {}".format(l2_reg))
#     regret_all_seeds = []
#     reward_all_seeds = []       

#     for seed in range(10):
#         print('Seed {}'.format(seed))
#         env = EnvMushroom(data_reduced)
#         replay_buffer = ReplayBuffer(10000)
#         net = BayesianNetwork(n_features, device, l2_reg=l2_reg, sigma=0.2)
#         optimizer = optim.Adam(net.parameters(), lr=1e-3)

#         regret_cumul = 0
#         actions = []
#         regrets = []
#         reward_cumul = 0
#         rewards = []
#         n_samples = 0
#         for frame_idx in tqdm(range(1, NUM_FRAMES + 1)):
#             X = env.sample()
#             n_samples += 1
#             action, incertitude = act(net, X, SAMPLES)
#             actions.append(action)
#             reward = env.hit(action)
#             regret_cumul += env.regret(action)
#             regrets.append(regret_cumul)

#             reward_cumul += reward
#             rewards.append(reward_cumul)
#             replay_buffer.push(X, action, reward)
            
#             if frame_idx % 10 == 0:
#                 if len(replay_buffer.buffer) > 2:
#                     compute_loss(net, optimizer, replay_buffer, n_samples, SAMPLES, BATCH_SIZE)

#         regret_all_seeds.append(regrets)
#         reward_all_seeds.append(rewards)
#         # pickle.dump(regret_all_seeds, open('data/regrets_bayes.p', 'wb'))
#         # print('Dump regrets : Done')
#         # pickle.dump(reward_all_seeds, (open('data/rewards_bayes.p', 'wb')))
#         # print('Dump rewards : Done')

#     regret_dico[l2_reg] = regret_all_seeds
#     reward_dico[l2_reg] = reward_all_seeds

#     pickle.dump(regret_dico, open('data/regret_dico_l2_reg.p', 'wb'))
#     pickle.dump(reward_dico, (open('data/reward_dico_l2_reg.p', 'wb')))
