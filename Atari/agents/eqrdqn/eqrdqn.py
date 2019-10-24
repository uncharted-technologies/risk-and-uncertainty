

import random
import time
import numpy as np

import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim

import pprint as pprint

from agents.common.replay_buffer import ReplayBuffer
from agents.common.logger import Logger
from agents.common.utils import set_global_seed
from agents.common.utils import quantile_huber_loss

class EQRDQN():
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
        adam_epsilon=1e-8,
        logging=False,
        log_folder_details=None,
        seed=None,
        notes=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        self.logger = []
        self.timestep=0
        self.log_folder_details = log_folder_details

        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = random.randint(0, 1e6) if seed is None else seed
        self.logger=None

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

        self.train_parameters = {'Notes':notes,
                'env':env.unwrapped.spec.id,
                'network':str(self.network),
                'replay_start_size':replay_start_size,
                'replay_buffer_size':replay_buffer_size,
                'gamma':gamma,
                'update_target_frequency':update_target_frequency,
                'minibatch_size':minibatch_size,
                'learning_rate':learning_rate,
                'update_frequency':update_frequency,
                'kappa':kappa,
                'n_quantiles':n_quantiles,
                'weight_scale':self.network.weight_scale,
                'prior':prior,
                'adam_epsilon':adam_epsilon,
                'seed':self.seed}


        self.n_greedy_actions = 0

    def learn(self, timesteps, verbose=False):

        self.train_parameters['train_steps'] = timesteps
        pprint.pprint(self.train_parameters)

        if self.logging:
            self.logger = Logger(self.log_folder_details,self.train_parameters)

        # Initialize the state
        state = torch.as_tensor(self.env.reset())
        this_episode_time = 0
        score = 0
        t1 = time.time()

        for timestep in range(timesteps):

            is_training_ready = timestep >= self.replay_start_size

            # Select action
            action = self.act(state.to(self.device).float(), thompson_sampling=True)

            # Perform action in environments
            state_next, reward, done, _ = self.env.step(action)

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            done = torch.as_tensor([done], dtype=torch.float)
            state_next = torch.as_tensor(state_next)
            self.replay_buffer.add(state, action, reward, state_next, done)

            score += reward.item()
            this_episode_time += 1

            if done:
    
                if verbose:
                    print("Timestep : {}, score : {}, Time : {} s".format(timestep, score, round(time.time() - t1, 3)))
                if self.logging:
                    self.logger.add_scalar('Episode_score', score, timestep)
                    self.logger.add_scalar('Non_greedy_fraction', 1-self.n_greedy_actions/this_episode_time, timestep)
                state = torch.as_tensor(self.env.reset())
                score = 0
                if self.logging:
                    self.logger.add_scalar('Q_at_start', self.get_max_q(state.to(self.device).float()), timestep)
                t1 = time.time()
                self.n_greedy_actions = 0
                this_episode_time = 0
            else:
                state = state_next

            if is_training_ready:

                # Update main network
                if timestep % self.update_frequency == 0:

                    # Sample a batch of transitions
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    # Train on selected batch
                    loss = self.train_step(transitions)
                    if self.logging and timesteps < 1000000:
                        self.logger.add_scalar('Loss', loss, timestep)

                # Update target Q
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if (timestep+1) % 250000 == 0:
                self.save(timestep=timestep+1)

            self.timestep=timestep


        if self.logging:
            self.logger.save()
            self.save()
        
    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions

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

    def act(self, state, thompson_sampling=False):

        action = self.predict(state,thompson_sampling=thompson_sampling)
        
        return action

    @torch.no_grad()
    def predict(self, state, thompson_sampling=False):
        if not thompson_sampling:
            net1,net2 = self.network(state)
            net1 = net1.view(self.env.action_space.n,self.n_quantiles)
            net2 = net2.view(self.env.action_space.n,self.n_quantiles)
            action_means = torch.mean((net1+net2)/2,dim=1)
            action = action_means.argmax().item()
        else:
            net1,net2 = self.network(state)
            net1_target,net2_target = self.target_network(state)
            net1 = net1.view(self.env.action_space.n,self.n_quantiles)
            net2 = net2.view(self.env.action_space.n,self.n_quantiles)
            net1_target = net1_target.view(self.env.action_space.n,self.n_quantiles)
            net2_target = net2_target.view(self.env.action_space.n,self.n_quantiles)
            action_means = torch.mean((net1+net2)/2,dim=1)
            action_uncertainties = torch.mean((net1_target-net2_target)**2,dim=1)/2
            samples = torch.distributions.multivariate_normal.MultivariateNormal(action_means,covariance_matrix=torch.diagflat(action_uncertainties)).sample()
            action = samples.argmax().item()
            if action == action_means.argmax().item():
                self.n_greedy_actions += 1

        return action

    @torch.no_grad()
    def get_max_q(self,state):
        net1,net2 = self.network(state)
        net1 = net1.view(self.env.action_space.n,self.n_quantiles)
        net2 = net2.view(self.env.action_space.n,self.n_quantiles)
        action_means = torch.mean((net1+net2)/2,dim=1)
        max_q = action_means.max().item()
        return max_q


    def save(self,timestep=None):
        if not self.logging:
            raise NotImplementedError('Cannot save without log folder.')

        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'

        save_path = self.logger.log_folder + '/' + filename

        torch.save(self.network.state_dict(), save_path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location='cpu'))
        self.target_network.load_state_dict(torch.load(path,map_location='cpu'))