#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements the QR-DQN algorithm as in Dabney, Will, et al. "Distributional reinforcement learning with quantile regression."
Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

"""

import random
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from logger import Logger
from utils import set_global_seed

from utils import quantile_huber_loss

class QRDQN():
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
        learning_rate=1e-4,
        update_frequency=1,
        prior=0.01,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        final_exploration_step=1000000,
        adam_epsilon=1e-8,
        logging=False,
        log_folder=None,
        seed=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.initial_exploration_rate = initial_exploration_rate
        self.epsilon = self.initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        self.logger = []
        self.timestep=0
        self.log_folder = log_folder

        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = random.randint(0, 1e6) if seed is None else seed
        set_global_seed(self.seed, self.env)

        self.n_quantiles = n_quantiles

        self.network = network(self.env.observation_space, self.env.action_space.n*self.n_quantiles, self.env.action_space.n*self.n_quantiles).to(self.device)
        self.target_network = network(self.env.observation_space, self.env.action_space.n*self.n_quantiles, self.env.action_space.n*self.n_quantiles).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

        self.anchor1 = [p.data.clone() for p in list(self.network.output_1.parameters())]
        self.anchor2 = [p.data.clone() for p in list(self.network.output_2.parameters())]

        self.loss = quantile_huber_loss
        self.kappa = kappa
        self.prior = prior

    def learn(self, timesteps, verbose=False):

        if self.logging:
            self.logger = Logger(self.log_folder)

        # Initialize state
        state = torch.as_tensor(self.env.reset())
        score = 0
        t1 = time.time()

        for timestep in range(timesteps):

            is_training_ready = timestep >= self.replay_start_size

            # Choose action
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)

            # Execute action in environment
            state_next, reward, done, _ = self.env.step(action)

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            done = torch.as_tensor([done], dtype=torch.float)
            state_next = torch.as_tensor(state_next)
            self.replay_buffer.add(state, action, reward, state_next, done)

            score += reward.item()

            if done:
                # Reinitialize the state
                if verbose:
                    print("Timestep : {}, score : {}, Time : {} s".format(timestep, score, round(time.time() - t1, 3)))
                if self.logging:
                    self.logger.add_scalar('Episode_score', score, timestep)
                state = torch.as_tensor(self.env.reset())
                score = 0
                t1 = time.time()
            else:
                state = state_next

            if is_training_ready:

                # Update main network
                if timestep % self.update_frequency == 0:

                    # Sample batch of transitions
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    # Train on selected transitions
                    loss = self.train_step(transitions)
                    if self.logging:
                        self.logger.add_scalar('Loss', loss, timestep)

                # Update weights of target Q network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if (timestep+1) % 250000 == 0:
                self.save(timestep=timestep+1)

            self.timestep=timestep


        if self.logging:
            self.logger.save()
        
    def train_step(self, transitions):

        states, actions, rewards, states_next, dones = transitions

        # Calculate target Q
        with torch.no_grad():
            
            target1,target2 = self.target_network(states_next.float())
            target1 = target1.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles)
            target2 = target2.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles)

        best_action_idx = torch.mean((target1+target2)/2,dim=2).max(1, True)[1].unsqueeze(2)
        q_value_target = 0.5*target1.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))\
            + 0.5*target2.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))

        # Calculate TD target
        td_target = rewards.unsqueeze(2).repeat(1,1,self.n_quantiles) \
            + (1 - dones.unsqueeze(2).repeat(1,1,self.n_quantiles)) * self.gamma * q_value_target

        # Calculate Q
        out1,out2 = self.network(states.float())
        out1 = out1.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles) 
        out2 = out2.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles) 

        q_value1 = out1.gather(1, actions.unsqueeze(2).repeat(1,1,self.n_quantiles))
        q_value2 = out2.gather(1, actions.unsqueeze(2).repeat(1,1,self.n_quantiles))

        loss1 = self.loss(q_value1.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss2 = self.loss(q_value2.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        quantile_loss = loss1+loss2

        diff1=[]
        for i, p in enumerate(self.network.output_1.parameters()):
            diff1.append(torch.sum((p - self.anchor1[i])**2))

        diff2=[]
        for i, p in enumerate(self.network.output_2.parameters()):
            diff2.append(torch.sum((p - self.anchor2[i])**2))

        diff1 = torch.stack(diff1).sum()
        diff2 = torch.stack(diff2).sum()

        anchor_loss = self.prior*(diff1+diff2)

        loss = quantile_loss + anchor_loss

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):

        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            # Action that maximizes Q function
            action = self.predict(state)
        else:
            # Random action
            action = np.random.randint(0, self.env.action_space.n)
        return action

    @torch.no_grad()
    def predict(self, state):
        net1,net2 = self.network(state)
        net1 = net1.view(self.env.action_space.n,self.n_quantiles)
        net2 = net2.view(self.env.action_space.n,self.n_quantiles)
        action_means = torch.mean((net1+net2)/2,dim=1)
        action = action_means.argmax().item()

        return action

    def update_epsilon(self, timestep):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
            timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    def save(self,timestep=None):
        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'

        if self.log_folder is not None:
            save_path = self.log_folder + '/' +filename
        else:
            save_path = filename

        torch.save(self.network.state_dict(), save_path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location='cpu'))