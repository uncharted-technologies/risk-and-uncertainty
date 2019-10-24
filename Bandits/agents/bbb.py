import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from agents.common.logger import Logger
from agents.common.replay_buffer import ReplayBuffer
from agents.bbb_utils import BayesianNetwork


class BBBAgent():

    def __init__(self,
    env,
    network,
    mean_prior=0,
    std_prior=0.01,
    noise_scale=0.01,
    logging=True,
    train_freq=10,
    updates_per_train=100,
    batch_size=32,
    start_train_step=10,
    log_folder_details=None,
    learning_rate=1e-3,
    bayesian_sample_size = 20,
    verbose=False
    ):

        self.env = env
        self.network = BayesianNetwork(env.n_features,torch.device('cpu'),std_prior,noise_scale)
        self.logging = logging
        self.replay_buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.log_folder_details = log_folder_details
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-8)
        self.train_freq = train_freq
        self.start_train_step = start_train_step
        self.updates_per_train = updates_per_train
        self.bayesian_sample_size = bayesian_sample_size
        self.verbose=verbose
        
        self.n_samples = 0
        self.timestep = 0

        self.train_parameters = {
        'mean_prior':mean_prior,
        'std_prior':std_prior,
        'noise_scale':noise_scale,
        'train_freq':train_freq,
        'updates_per_train':updates_per_train,
        'batch_size':batch_size,
        'start_train_step':start_train_step,
        'learning_rate':learning_rate,
        'bayesian_sample_size':bayesian_sample_size
        }

    def learn(self,n_steps):

        self.train_parameters['n_steps']=n_steps

        if self.logging:
            self.logger = Logger(self.log_folder_details,self.train_parameters)

        cumulative_regret = 0

        for timestep in range(n_steps):

            x = self.env.sample()

            action, sampled_value = self.act(x.float())

            reward = self.env.hit(action)
            regret = self.env.regret(action)

            cumulative_regret += regret

            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)

            if action ==1:
                self.n_samples += 1
                self.replay_buffer.add(x, reward)

            if self.logging:
                self.logger.add_scalar('Cumulative_Regret', cumulative_regret, timestep)
                self.logger.add_scalar('Mushrooms_Eaten', self.n_samples, timestep)
                if self.env.y_sample == 1:
                    self.logger.add_scalar('Sampled_Value_Good', sampled_value.item(), self.timestep)
                else:
                    self.logger.add_scalar('Sampled_Value_Bad', sampled_value.item(), self.timestep)

            if timestep % self.train_freq == 0 and self.n_samples > self.start_train_step:

                if self.verbose:
                    print('Timestep: {}, cumulative regret {}'.format(str(timestep),str(cumulative_regret)))

                for update in range(self.updates_per_train):

                    samples = self.replay_buffer.sample(np.min([self.n_samples,self.batch_size]))
                    self.train_step(samples)

            self.timestep += 1

        if self.logging:
            self.logger.save()

    def train_step(self,samples):

        states, rewards = samples

        loss,_,_,_ = self.network.sample_elbo(states.float(),rewards,self.n_samples,np.min([self.n_samples,self.batch_size]),self.bayesian_sample_size)

        if self.logging:
            self.logger.add_scalar('Loss', loss.detach().item(), self.timestep)

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def act(self,x):

        action,sampled_value = self.predict(x)

        return action,sampled_value

    @torch.no_grad()
    def predict(self,x):

        sampled_value = self.network.forward(x)

        if sampled_value > 0:
            action = 1
        else:
            action = 0

        return action,sampled_value