
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents.qrdqn.qrdqn import QRDQN
from agents.common.networks.mlp import MLP

notes = "This is a test run"

env = gym.make("CartPole-v0")

nb_steps = 5000

agent = QRDQN( env,
                 MLP,
                 n_quantiles= 20,
                 kappa= 10,
                 replay_start_size=50,
                 replay_buffer_size=50000,
                 gamma=0.99,
                 update_target_frequency=50,
                 minibatch_size=32,
                 learning_rate=1e-3,
                 initial_exploration_rate=1,
                 final_exploration_rate=0.02,
                 final_exploration_step=5000,
                 adam_epsilon=1e-8,
                 update_frequency=1,
                 logging=True,
                 log_folder_details="Cartpole-QRDQN",
                 notes=notes)

agent.learn(timesteps=nb_steps, verbose=True)

scores = np.array(agent.logger.memory['Episode_score'])
plt.cla()
plt.plot(scores[:,1],scores[:,0])
plt.show()