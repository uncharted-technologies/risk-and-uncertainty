import torch
import numpy as np
import pandas as pd

from agents.egreedy import EGreedyAgent
from common.mushroom_env import MushroomEnv
from agents.common.qrnet import QRNet

NB_STEPS = 20000
N_SEEDS = 20

for i in range(N_SEEDS):
    
    env = MushroomEnv()
    agent = EGreedyAgent(env,
        QRNet,
        epsilon=0.05,
        n_quantiles=50,
        mean_prior=0,
        std_prior=0.1,
        logging=True,
        train_freq=1,
        updates_per_train=1,
        batch_size=32,
        start_train_step=32,
        log_folder_details='EGreedy',
        learning_rate=1e-2
        )

    agent.learn(NB_STEPS)