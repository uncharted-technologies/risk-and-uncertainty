"""
Code for Bayes by Backprop, modified from https://github.com/nitarshan/bayes-by-backprop 

"""

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 

class Gaussian(object):
    def __init__(self, mu, rho, device):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
        self.device = device
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.device)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, x):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixtureGaussian(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.gaussian = torch.distributions.Normal(0,sigma)
    
    def log_prob(self, x):
        return self.gaussian.log_prob(x).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, device, sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, sigma))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3 + np.log10(2), -2 + np.log10(2)))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, self.device)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, sigma))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3 + np.log10(2), -2 + np.log10(2)))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, self.device)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(sigma)
        self.bias_prior = ScaleMixtureGaussian(sigma)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x, sample=True):

        weight = self.weight.sample()
        bias = self.bias.sample()

        self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)

        return F.linear(x, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self, n_features, device, sigma=0.1, noise_scale=1):
        super().__init__()
        self.n_features = n_features
        self.device = device
        self.l1 = BayesianLinear(self.n_features, 100, self.device, sigma)
        self.l2 = BayesianLinear(100, 100, self.device, sigma)
        self.l3 = BayesianLinear(100, 1, self.device, sigma)
        self.noise_scale = noise_scale
        self.sigma = sigma

    def forward(self, x, sample=False):
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = self.l3(x, sample)
        return x
    
    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior
    
    def sample_elbo(self, X, reward, n_samples, batch_size, samples):
        outputs = torch.zeros(samples, batch_size).to(self.device)
        log_priors = torch.zeros(samples).to(self.device)
        log_variational_posteriors = torch.zeros(samples).to(self.device)
        for i in range(samples):
            outputs[i] = self.forward(X.squeeze(), sample=True).squeeze()
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.mse_loss(outputs.mean(dim=0), reward.squeeze(), reduction='mean')
        loss = self.noise_scale * (log_variational_posterior - log_prior) / (n_samples) + negative_log_likelihood

        return loss, log_prior, log_variational_posterior, negative_log_likelihood