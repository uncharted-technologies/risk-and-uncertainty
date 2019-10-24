
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents.eqrdqn.eqrdqn import EQRDQN
from agents.common.networks.mlp import MLP_Multihead


notes = "This is a test run"

env = gym.make("CartPole-v0")

nb_steps = 5000

agent = EQRDQN( env,
                 MLP_Multihead,
                 n_quantiles=20,
                 prior=0.01,
                 kappa=10,
                 replay_start_size=50,
                 replay_buffer_size=50000,
                 gamma=0.99,
                 update_target_frequency=50,
                 minibatch_size=32,
                 learning_rate=1e-3,
                 adam_epsilon=1e-8,
                 update_frequency=1,
                 logging=True,
                 log_folder_details="Cartpole-EQRDQN",
                 notes=notes)


agent.learn(timesteps=nb_steps, verbose=True)

scores = np.array(agent.logger.memory['Episode_score'])
plt.cla()
plt.plot(scores[:,1],scores[:,0])
plt.show()
