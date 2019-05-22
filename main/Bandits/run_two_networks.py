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

l2_reg = 10
anchored = True

n_networks = 20
n_seed = 10

dico_uncertainties = {}
dico_regrets = {}

for seed in range(n_seed):
    print("seed:", seed)

    current_model1 = QRContext(2, n_features, num_quant, mean_prior=mean_prior, std_prior=std_prior, l2_reg=l2_reg)
    current_model2 = QRContext(2, n_features, num_quant, mean_prior=mean_prior, std_prior=std_prior, l2_reg=l2_reg)
    target_model1  = QRContext(2, n_features, num_quant, mean_prior=mean_prior, std_prior=std_prior, l2_reg=l2_reg)
    target_model2  = QRContext(2, n_features, num_quant, mean_prior=mean_prior, std_prior=std_prior, l2_reg=l2_reg)


    optimizer1 = optim.Adam(current_model1.parameters(), lr=lr)
    optimizer2 = optim.Adam(current_model2.parameters(), lr=lr)

    _, incertitude, _, regret_cumul = play(data_reduced,
                                    batch_size,
                                    buffer_size,
                                    num_frames,
                                    current_model1,
                                    target_model1,
                                    optimizer1,
                                    current_model2,
                                    target_model2,
                                    optimizer2)

    dico_regrets[str(l2_reg)].append(regret_cumul)

pickle.dump(dico_regrets, open("regrets_anchored.p", "wb"))
print("Dump everything done")


