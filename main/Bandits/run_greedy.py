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

dico_uncertainties = {}
dico_regrets = {}

dico_regrets["eps_greedy"] = []

# Train eps greedy

for seed in range(10):
    print("seed:", seed)

    current_model_eps = QRDQN_epsilon(2, n_features, num_quant, mean_prior=mean_prior, std_prior=std_prior,
                                      l2_reg=0.005, epsilon=0.05)
    target_model_eps  = QRDQN_epsilon(2, n_features, num_quant, mean_prior=mean_prior, std_prior=std_prior,
                                    l2_reg=0.005, epsilon=0.05)

    optimizer_eps = optim.Adam(current_model_eps.parameters(), lr=lr)

    _, _, _, regret_cumul = play(data_reduced,
                                        batch_size,
                                        buffer_size,
                                        num_frames,
                                        current_model_eps,
                                        target_model_eps,
                                        optimizer_eps)

    dico_regrets["eps_greedy"].append(regret_cumul)

pickle.dump(dico_regrets, open("regrets_eps_greedy.p", "wb"))



