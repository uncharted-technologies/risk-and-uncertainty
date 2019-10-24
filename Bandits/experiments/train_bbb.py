import torch
import numpy as np
import pandas as pd

from agents.bbb import BBBAgent
from common.mushroom_env import MushroomEnv

NB_STEPS = 20000
N_SEEDS = 20

for i in range(N_SEEDS):
    
    env = MushroomEnv()
    agent = BBBAgent(env,
        None,
        mean_prior=0,
        std_prior=0.1,
        logging=True,
        train_freq=1,
        updates_per_train=1,
        batch_size=128,
        start_train_step=32,
        log_folder_details='BBB',
        learning_rate=1e-2,
        noise_scale=0.01,
        bayesian_sample_size = 2,
        verbose=True
        )

    agent.learn(NB_STEPS)