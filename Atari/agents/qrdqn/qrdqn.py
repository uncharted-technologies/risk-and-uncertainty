

import random
import time

from datetime import datetime
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import pprint as pprint

from agents.common.replay_buffer import ReplayBuffer
from agents.common.logger import Logger
from agents.common.utils import set_global_seed

from agents.dqn.dqn import DQN
from agents.common.utils import quantile_huber_loss

class QRDQN(DQN):
    def __init__(
        self,
        env,
        network,
        n_quantiles=50,
        kappa=1,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        minibatch_size=32,
        learning_rate=1e-3,
        update_frequency=1,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        final_exploration_step=1000000,
        adam_epsilon=1e-8,
        logging=False,
        log_folder_details=None,
        seed=None,
        loss="huber",
        notes=None
    ):

        super().__init__(env,network,replay_start_size,replay_buffer_size,gamma,
            update_target_frequency,minibatch_size,learning_rate,update_frequency,initial_exploration_rate,
            final_exploration_rate,final_exploration_step,adam_epsilon,logging,log_folder_details,seed,loss,notes)

        self.n_quantiles = n_quantiles
        self.network = network(self.env.observation_space, self.env.action_space.n*self.n_quantiles).to(self.device)
        self.target_network = network(self.env.observation_space, self.env.action_space.n*self.n_quantiles).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

        self.loss = quantile_huber_loss
        self.kappa = kappa

        self.train_parameters = {'Notes':notes,
                'env':env.unwrapped.spec.id,
                'network':str(self.network),
                'n_quantiles':n_quantiles,
                'kappa':kappa,
                'replay_start_size':replay_start_size,
                'replay_buffer_size':replay_buffer_size,
                'gamma':gamma,
                'update_target_frequency':update_target_frequency,
                'weight_scale':self.network.weight_scale,
                'minibatch_size':minibatch_size,
                'learning_rate':learning_rate,
                'update_frequency':update_frequency,
                'initial_exploration_rate':initial_exploration_rate,
                'final_exploration_rate':final_exploration_rate,
                'final_exploration_step':final_exploration_step,
                'adam_epsilon':adam_epsilon,
                'seed':self.seed}
        
    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions

        with torch.no_grad():
            target_outputs = self.target_network(states_next.float()).view(self.minibatch_size,self.env.action_space.n,self.n_quantiles)
            best_action_idx = torch.mean(target_outputs,dim=2).max(1, True)[1].unsqueeze(2)
            q_value_target = target_outputs.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))

        # Calculate TD Target
        td_target = rewards.unsqueeze(2).repeat(1,1,self.n_quantiles) + (1 - dones.unsqueeze(2).repeat(1,1,self.n_quantiles)) * self.gamma * q_value_target

        outputs = self.network(states.float()).view(self.minibatch_size,self.env.action_space.n,self.n_quantiles) 
        q_value = outputs.gather(1, actions.unsqueeze(2).repeat(1,1,self.n_quantiles))

        loss = self.loss(q_value.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def get_max_q(self,state):
        net = self.network(state)
        action_means = net.view(self.env.action_space.n,self.n_quantiles).mean(dim=1)
        max_q = action_means.max().item()
        return max_q

    @torch.no_grad()
    def predict(self, state):
        action = torch.mean(self.network(state).view(self.env.action_space.n,self.n_quantiles),dim=1).argmax().item()
        return action

