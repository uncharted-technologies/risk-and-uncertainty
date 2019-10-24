import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from agents.common.logger import Logger
from agents.common.replay_buffer import ReplayBuffer
from agents.common.utils import quantile_huber_loss

class ThompsonAgent():

    def __init__(self,env,
    network,
    n_quantiles=20,
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
    verbose=False
    ):

        self.env = env
        self.network1 = network(env.n_features, n_quantiles, mean_prior, std_prior)
        self.network2 = network(env.n_features, n_quantiles, mean_prior, std_prior)
        self.optimizer = optim.Adam(list(self.network1.parameters()) + list(self.network2.parameters()), lr=learning_rate, eps=1e-8)

        self.logging = logging
        self.replay_buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.log_folder_details = log_folder_details
        self.n_quantiles = n_quantiles
        self.train_freq = train_freq
        self.start_train_step = start_train_step
        self.updates_per_train = updates_per_train
        self.n_samples = 0
        self.noise_scale = noise_scale
        self.std_prior = std_prior
        self.verbose = verbose

        self.prior1 = [p.data.clone() for p in list(self.network1.features.parameters())]
        self.prior2 = [p.data.clone() for p in list(self.network2.features.parameters())]

        self.train_parameters = {'n_quantiles':n_quantiles,
        'mean_prior':mean_prior,
        'std_prior':std_prior,
        'train_freq':train_freq,
        'updates_per_train':updates_per_train,
        'batch_size':batch_size,
        'start_train_step':start_train_step,
        'learning_rate':learning_rate,
        'noise_scale':noise_scale
        }

    def learn(self,n_steps):

        self.train_parameters['n_steps']=n_steps

        if self.logging:
            self.logger = Logger(self.log_folder_details,self.train_parameters)

        cumulative_regret = 0

        for timestep in range(n_steps):

            x = self.env.sample()
            

            action,uncertainty, sampled_value = self.act(x.float())

            reward = self.env.hit(action)
            regret = self.env.regret(action)

            cumulative_regret += regret

            reward = torch.as_tensor([reward], dtype=torch.float)

            if self.logging:
                self.logger.add_scalar('Cumulative_Regret', cumulative_regret, timestep)
                self.logger.add_scalar('Mushrooms_Eaten', self.n_samples, timestep)
                if self.env.y_sample == 1:
                    self.logger.add_scalar('Uncertainty_Good', uncertainty, timestep)
                    self.logger.add_scalar('Sampled_Value_Good', sampled_value, timestep)
                else:
                    self.logger.add_scalar('Uncertainty_Bad', uncertainty, timestep)
                    self.logger.add_scalar('Sampled_Value_Bad', sampled_value, timestep)

            if action == 1:
                self.replay_buffer.add(x, reward)
                self.n_samples += 1

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

        q_value1 = self.network1(states.float()).view(np.min([self.n_samples,self.batch_size]),self.n_quantiles) 
        q_value2 = self.network2(states.float()).view(np.min([self.n_samples,self.batch_size]),self.n_quantiles) 

        loss1 = quantile_huber_loss(q_value1.squeeze(), target.squeeze())
        loss2 = quantile_huber_loss(q_value2.squeeze(), target.squeeze())

        reg = []
        for i, p in enumerate(self.network1.features.parameters()):
            diff = (p - self.prior1[i])
            reg.append(torch.sum(diff**2))
        loss_anchored1 = torch.sum(torch.stack(reg))

        reg = []
        for i, p in enumerate(self.network2.features.parameters()):
            diff = (p - self.prior2[i])
            reg.append(torch.sum(diff**2))
        loss_anchored2 = torch.sum(torch.stack(reg))

        loss = loss1 + loss2 + self.noise_scale*(loss_anchored1+loss_anchored2)/(self.std_prior**2*self.n_samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def act(self,x):

        action,uncertainty,sampled_value = self.predict(x)

        return action,uncertainty,sampled_value

    @torch.no_grad()
    def predict(self,x):

        net1 = self.network1(x)
        net2 = self.network2(x)

        action_mean = torch.mean((net1+net2)/2)
        action_uncertainty = torch.sqrt(torch.mean((net1-net2)**2)/2)
        sampled_value = torch.distributions.Normal(action_mean,action_uncertainty).sample()

        if sampled_value > 0:
            action = 1
        else:
            action = 0

        return action, action_uncertainty.item(), sampled_value