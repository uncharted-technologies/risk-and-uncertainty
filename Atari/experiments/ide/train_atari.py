
from agents.common.networks import CNNDeepmind_Multihead
from agents.ide.ide import IDE
from agents.common.atari_wrappers import make_atari, wrap_deepmind

import pickle
import numpy as np
import matplotlib.pyplot as plt

notes="This is a test run"

env_name = "BreakoutNoFrameskip-v4"
env = make_atari(env_name,noop=True)
env = wrap_deepmind(env, episode_life=True)

nb_steps = 12500000

agent = IDE( env,
                 CNNDeepmind_Multihead,
                 n_quantiles=200,
                 kappa=0,
                 prior = 0.0001,
                 replay_start_size=50000,
                 replay_buffer_size=1000000,
                 gamma=0.99,
                 update_target_frequency=10000,
                 minibatch_size=32,
                 learning_rate=5e-5,
                 adam_epsilon=0.01/32,
                 update_frequency=4,
                 log_folder_details=env_name+'-IDE',
                 logging=True,
                 notes=notes)


agent.learn(timesteps=nb_steps, verbose=True)
