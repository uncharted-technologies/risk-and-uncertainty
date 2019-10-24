import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from agents.common.logger import Logger
from agents.common.replay_buffer import ReplayBuffer
from agents.common.utils import quantile_huber_loss

class EGreedyAgent():

    def __init__(self,env,
    network,
    epsilon=0.05,
    n_quantiles=20,
    mean_prior=0,
    std_prior=0.01,
    logging=True,
    train_freq=10,
    updates_per_train=100,
    batch_size=32,
    start_train_step=10,
    log_folder_details=None,
    learning_rate=1e-3,
    verbose=False
    ):

        self.env = env
        self.network = network(env.n_features, n_quantiles, mean_prior, std_prior)
        self.logging = logging
        self.replay_buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.log_folder_details = log_folder_details
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-8)
        self.n_quantiles = n_quantiles
        self.train_freq = train_freq
        self.start_train_step = start_train_step
        self.updates_per_train = updates_per_train
        self.verbose = verbose
        
        self.n_samples = 0


        self.train_parameters = {'epsilon':epsilon,
        'n_quantiles':n_quantiles,
        'mean_prior':mean_prior,
        'std_prior':std_prior,
        'train_freq':train_freq,
        'updates_per_train':updates_per_train,
        'batch_size':batch_size,
        'start_train_step':start_train_step,
        'learning_rate':learning_rate
        }

    def learn(self,n_steps):

        self.train_parameters['n_steps']=n_steps

        if self.logging:
            self.logger = Logger(self.log_folder_details,self.train_parameters)

        cumulative_regret = 0

        for timestep in range(n_steps):

            x = self.env.sample()

            action = self.act(x.float())

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

            if timestep % self.train_freq == 0 and self.n_samples > self.start_train_step:

                if self.verbose:
                    print('Timestep: {}, cumulative regret {}'.format(str(timestep),str(cumulative_regret)))

                for update in range(self.updates_per_train):

                    samples = self.replay_buffer.sample(np.min([self.n_samples,self.batch_size]))
                    self.train_step(samples)

        if self.logging:
            self.logger.save()

    def train_step(self,samples):

        states, rewards = samples

        target = rewards.repeat(1,self.n_quantiles)

        q_value = self.network(states.float()).view(np.min([self.n_samples,self.batch_size]),self.n_quantiles) 

        loss = quantile_huber_loss(q_value.squeeze(), target.squeeze())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def act(self,x):

        if np.random.uniform() >= self.epsilon:
            action = self.predict(x)
        else:
            action = np.random.randint(0, 2)
        return action

    @torch.no_grad()
    def predict(self,x):

        estimated_value = torch.mean(self.network(x))

        if estimated_value > 0:
            action = 1
        else:
            action = 0

        return action