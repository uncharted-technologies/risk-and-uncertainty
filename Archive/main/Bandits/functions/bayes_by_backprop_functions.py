# Code inspired by the github : https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 

# Params to set the priors in the Bayesian Linear
SIGMA = torch.FloatTensor([0.2])
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)


class Gaussian(object):
    def __init__(self, mu, rho, device):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.device = device
        self.normal = torch.distributions.Normal(0,1)
        # self.sigma = torch.FloatTensor([0.01])
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.device)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixtureGaussian(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.gaussian = torch.distributions.Normal(0,sigma)
    
    def log_prob(self, input):
        prob = torch.exp(self.gaussian.log_prob(input))
        return torch.log(prob).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, device, sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.1, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3 + np.log10(2), -2 + np.log10(2)))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, self.device)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.1, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3 + np.log10(2), -2 + np.log10(2)))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, self.device)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(sigma)
        self.bias_prior = ScaleMixtureGaussian(sigma)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self, n_features, device, sigma=0.1, l2_reg=1):
        super().__init__()
        self.n_features = n_features
        self.device = device
        self.l1 = BayesianLinear(self.n_features, 100, self.device, sigma)
        self.l2 = BayesianLinear(100, 100, self.device, sigma)
        self.l3 = BayesianLinear(100, 2, self.device, sigma)
        self.l2_reg = l2_reg
        self.sigma = sigma

    def forward(self, x, sample=False):
        x = x.view(-1, self.n_features)
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
    
    def sample_elbo(self, X, action, reward, n_samples, batch_size, samples):
        action = action.unsqueeze(1)
        outputs = torch.zeros(samples, batch_size).to(self.device)
        log_priors = torch.zeros(samples).to(self.device)
        log_variational_posteriors = torch.zeros(samples).to(self.device)
        for i in range(samples):
            output = self(X, sample=True)
            outputs[i] = output.gather(1, action).squeeze(1)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.mse_loss(outputs.mean(0), reward, reduction='sum')
        loss = self.l2_reg * (log_variational_posterior - log_prior) / n_samples + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

def act(net, X, samples):
    outputs = [net(X.float(), sample=True) for i in range(samples * 10)]
    outputs = torch.stack(outputs).squeeze().detach()
    mean = outputs.mean(0)
    uncertainty = outputs.std(0)
    Q = [np.random.normal(loc=mean[k].item(), scale=i.abs().item()) for k,i in enumerate(uncertainty)]
    Q = torch.tensor(Q)
    _, a_star = torch.max(Q, 0)
    return a_star.item(), uncertainty.detach()


def train_step(net, optimizer, X, action, reward, n_samples, samples):
    net.train()
    
    net.zero_grad()
    loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(X,
                                                                                          action,
                                                                                          reward,
                                                                                          n_samples= n_samples,
                                                                                          batch_size = X.size(0),
                                                                                          samples=samples)
    loss.backward()
    optimizer.step()

    return loss, log_prior, log_variational_posterior, negative_log_likelihood


def compute_loss(net, optimizer, replay_buffer, n_samples, samples, batch_size, verbose=False):
    for i in range(100):
        if len(replay_buffer.buffer) < batch_size:
            X, action, reward = replay_buffer.sample_all()

        else:
            X, action, reward = replay_buffer.sample_batch(batch_size)


        X          = Variable(torch.FloatTensor(np.float32(X)))
        action     = Variable(torch.LongTensor(action))
        reward     = torch.FloatTensor(reward)

        loss, log_prior, log_variational_posterior, negative_log_likelihood = train_step(net, optimizer, X, action, reward, n_samples, samples)
        if verbose:
            print("Loss: {}".format(loss))