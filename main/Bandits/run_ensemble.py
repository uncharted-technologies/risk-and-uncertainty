import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import copy
from scipy.stats import norm
import random
from tqdm import tqdm
import pickle

from functions.eqrdqn_functions import *

d = defaultdict(LabelEncoder)
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
data_reduced = pd.read_csv("data/data_mushrooms.csv", index_col=0)


#Fixed param

n_features = data_reduced.shape[1] - 1
num_quant = 50
mean_prior = 0
std_prior = .1
lr = 1e-3

batch_size = 32
buffer_size = 20000
num_frames = 20000

anchored = True

n_networks = 20
n_seed = 10

dico_uncertainties = {}

l2_reg = 0.005 # Because we only have one output 
regret_20 = []

for seed in range(n_seed):
    print("Seed: {}".format(seed))
    # Initialization

    current_models = []
    target_models = []
    optimizers = []

    for i in range(n_networks):
        current_models.append(QRContext(2,
                                        n_features,
                                        1,
                                        mean_prior=mean_prior,
                                        std_prior=std_prior,
                                        l2_reg=l2_reg))
        
        target_models.append(QRContext(2,
                                    n_features,
                                    1,
                                    mean_prior=mean_prior,
                                    std_prior=std_prior,
                                    l2_reg=l2_reg))
        
        optimizers.append(optim.Adam(current_models[-1].parameters(), lr=lr))

    # Play

    # Set environment
    env = EnvMushroom(data_reduced)
    replay_buffer = ReplayBuffer(buffer_size)
    n_samples = 0

    all_reward = 0
    reward_list = []
    reward_cumul = []
    actions = []

    incertitudes = []

    all_regret = 0
    regret_cumul = []

    for frame_idx in tqdm(range(1, num_frames + 1)):
        X = env.sample()
        n_samples += 1
        dists = [net(X).squeeze() for net in current_models]
        action, incertitude = choose_action_Thompson_ensemble_non_distributed(dists)
        incertitudes.append(incertitude)
        action = action.item()


        actions.append(action)


        reward = env.hit(action)
        reward_list.append(reward)
        all_reward += reward
        reward_cumul.append(all_reward)

        all_regret += env.regret(action)
        regret_cumul.append(all_regret)

        replay_buffer.push(X, action, reward)
        
        if frame_idx % 10 == 0:
            if len(replay_buffer.buffer) > 2:
                
                compute_td_loss_ensemble(
                        current_models,
                        target_models,
                        optimizers,
                        replay_buffer,
                        1,
                        n_samples,
                        ensemble_size=len(current_models),
                        batch_size=batch_size,
                        anchored=True)


        # Update target
        if frame_idx % (num_frames/10) == 0:
            for j in range(len(current_models)):
                update_target(current_models[j], target_models[j])

    regret_20.append(regret_cumul)

    pickle.dump(regret_20, open('data/regret_20_nets_non_distributed.p', 'wb'))
    print("20 nets non distributed dumped")
