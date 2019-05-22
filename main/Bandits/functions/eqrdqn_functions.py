import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import copy
from scipy.stats import norm
import random
from tqdm import tqdm
import pickle

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)


class EnvMushroom:
    """
    Creates a mushroom environment to sample a mushroom and give a reward according the picked action
    """
    def __init__(self, data_encoded):
        self.y = torch.from_numpy(data_encoded["target"].values).reshape(-1, 1) * (-2) + 1 # Set y in {-1, 1}
        self.X = torch.from_numpy(data_encoded.drop("target", 1).values)
        self.n_samples = len(data_encoded)
        self.good_mushroom = (1, 1)
        self.bad_mushroom = (-3, 1)
        self.y_hist = []
        
    def sample(self, batch_size=1):
        """
        Sample one (X, y)
        """
        if batch_size == 1:
            ix = np.random.randint(0, self.n_samples)
            self.y_sample = self.y[ix].item()
            self.y_hist.append(self.y_sample)
            return self.X[ix, :].unsqueeze(0)
        else:   
            ix = np.random.randint(0, self.n_samples, batch_size)
            self.y_sample = self.y[ix].reshape(-1).numpy()
            self.y_hist += list(self.y_sample)
            return self.X[ix, :]
    
    def clear_y_hist(self):
        self.y_hist = []
    
    
    def hit(self, action):
        if action == 0:
            return 0
        else:
            if self.y_sample == 1:
                return np.random.normal(self.good_mushroom[0], self.good_mushroom[1])
            else:
                return np.random.normal(self.bad_mushroom[0], self.bad_mushroom[1])
    

    def regret(self, action):
        if self.y_sample == 1:
            return self.good_mushroom[0] * (1 - action)
        else:
            return action * self.bad_mushroom[0] * (-1)


class ReplayBuffer:
    """
    Replay buffer to gather new experiences and sample batches for to update the agent
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_red = []
        
    def push(self, X, action, reward):
        self.buffer.append((X.squeeze(0), action, reward))
    
    
    def sample_batch(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        l = list(map(list, zip(*samples)))
        return torch.stack(l[0]), l[1], l[2]
    
    def sample_all(self):
        l = list(map(list, zip(*self.buffer)))
        return torch.stack(l[0]), l[1], l[2]

def init_weights(m, mean, std):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean, std)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean, std)

        
class QRContext(nn.Module):
    """
    The quantile estimator modelized by a NN.
    Initialized with respect to a normal prior (mean_prior, std_prior)
    """
    def __init__(self, num_actions, n_features, num_quants, mean_prior, std_prior, l2_reg):
        super(QRContext, self).__init__()
        self.num_actions = num_actions
        self.n_features = n_features
        self.num_quants  = num_quants
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        self.l2_reg = l2_reg
        self.name = "QRDQN"
        
        self.features = nn.Sequential(
            nn.Linear(self.n_features, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_actions * self.num_quants, bias=False))

        self.features.apply(lambda m: init_weights(m, self.mean_prior, self.std_prior))
        #self.features[0].requires_grad = False
        # Save prior 
        self.prior = [p.data.clone() for p in list(self.features.parameters())]


    def forward(self, x):
        batch_size = x.size(0)
        x = Variable(torch.FloatTensor(np.float32(x)))
        x = self.features(x)
        x = x.view(batch_size, self.num_actions, self.num_quants)
        return x


class QRDQN_epsilon(nn.Module):
    """
    The quantile estimator modelized by a NN.
    Initialized with respect to a gaussian prior (mean_prior, std_prior)
    """
    def __init__(self, num_actions, n_features, num_quants, mean_prior, std_prior, epsilon, l2_reg):
        super(QRDQN_epsilon, self).__init__()
        self.num_actions = num_actions
        self.n_features = n_features
        self.num_quants  = num_quants
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        self.epsilon = epsilon
        self.l2_reg = l2_reg
        self.name = "QRDQN_epsilon"
        
        self.features = nn.Sequential(
            nn.Linear(self.n_features, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_actions * self.num_quants))

        self.features.apply(lambda m: init_weights(m, self.mean_prior, self.std_prior))
        # Save prior 
        self.prior = [p.data.clone() for p in list(self.features.parameters())]


    def forward(self, x):
        batch_size = x.size(0)
        x = Variable(torch.FloatTensor(np.float32(x)))
        x = self.features(x)
        x = x.view(batch_size, self.num_actions, self.num_quants)
        return x

    
    def choose_action(self, state):
        if random.random() > self.epsilon:
            qvalues = self.forward(state).mean(2)
            action  = qvalues.max(1)[1]
        else:
            action = torch.randint(self.num_actions,(state.size(0),), dtype=torch.int64)
        return action


class QRDQN_dropout(nn.Module):
    """
    The quantile estimator modelized by a NN.
    Initialized with respect to a gaussian prior (mean_prior, std_prior)
    """
    def __init__(self, num_actions, n_features, num_quants, p, mean_prior, std_prior, l2_reg):
        super(QRDQN_dropout, self).__init__()
        self.num_actions = num_actions
        self.n_features = n_features
        self.num_quants  = num_quants
        self.p = p
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        self.l2_reg = l2_reg
        self.name = "QRDQN_dropout"
        
        self.features = nn.Sequential(
            nn.Linear(self.n_features, 100),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(100, self.num_actions * self.num_quants))

        self.features.apply(lambda m: init_weights(m, self.mean_prior, self.std_prior))
        # Save prior 
        self.prior = [p.data.clone() for p in list(self.features.parameters())]


    def forward(self, x):
        batch_size = x.size(0)
        x = Variable(torch.FloatTensor(np.float32(x)))
        x = self.features(x)
        x = x.view(batch_size, self.num_actions, self.num_quants)
        return x


def choose_action_Thompson(dist1, dist2, num_quant):
    incertitude = torch.sum((dist1.squeeze(0) - dist2.squeeze(0)) ** 2, dim=1)
    incertitude /= num_quant
    incertitude = torch.sqrt(incertitude)
    
    mean_a = 0.5 * (dist1.mean(2).squeeze(0) + dist2.mean(2).squeeze(0))
    
    Q = [np.random.normal(loc=mean_a[k].item(),
                                   scale=i.item()) for k, i in enumerate(incertitude.detach())]  
    
    Q = torch.tensor(Q)
    _, a_star = torch.max(Q, 0)
    
    return a_star, incertitude.detach()


def projection_distribution(X, dist, reward, target_model, num_quant):

#     next_dist = target_model(X)
#     next_action = next_dist.mean(2).max(1)[1]
#     next_action = next_action.unsqueeze(1).unsqueeze(1).expand(1, 1, num_quant)
#     next_dist = next_dist.gather(1, next_action).squeeze(1).cpu().data
    batch_size = X.size(0)
    expected_quant = reward.unsqueeze(1) # + 0. * next_dist
    expected_quant = Variable(expected_quant)

    quant_idx = torch.sort(dist, 1, descending=False)[1]

    tau_hat = torch.linspace(0.0, 1.0 - 1./num_quant, num_quant) + 0.5 / num_quant
    tau_hat = tau_hat.unsqueeze(0).repeat(batch_size, 1)
#     quant_idx = quant_idx.cpu().data
#     batch_idx = np.arange(batch_size)
#     tau = tau_hat[:, quant_idx][batch_idx, batch_idx]
#     tau = tau_hat.unsqueeze(1).repeat(1,2,1)
        
    return tau_hat, expected_quant        


def train_step(current_model, target_model, optimizer, X, action, reward, n_samples, num_quant, anchored, k=0):
    dist = current_model(X)
    action = action.unsqueeze(1).unsqueeze(1).expand(action.shape[0], 1, num_quant)
    dist = dist.gather(1, action).squeeze(1)   

    tau, expected_quant = projection_distribution(X, dist, reward, target_model, num_quant)

    u = expected_quant - dist

    # Update network 1

    if k != 0:
        huber_loss = 0.5 * u.abs().clamp(min=0.0, max=k).pow(2)
        huber_loss += k * (u.abs() -  u.abs().clamp(min=0.0, max=k))
    else:
        huber_loss = u.abs()
        
    quantile_loss = (tau - (u < 0).float()).abs() * huber_loss
    loss = quantile_loss.mean(0).sum()

    if anchored == True:
        reg = []
        for i, p in enumerate(current_model.parameters()):
            sigma = torch.tensor([current_model.l2_reg]).repeat(p.shape[0])
            sigma = torch.diag(sigma).float()
            diff = (p - current_model.prior[i]).view(p.shape[0], -1)
            reg.append(torch.norm(torch.mm(sigma, diff), 2)**2)
        loss += torch.sum(torch.stack(reg)) / n_samples
#             if n_samples % 100 == 0:
#                 print(n_samples, torch.sum(torch.stack(reg)) / n_samples /loss)
    else:
        reg = []
        for i, p in enumerate(current_model.parameters()):
            sigma = torch.tensor([current_model.l2_reg]).repeat(p.shape[0])
            sigma = torch.diag(sigma).float()
            diff = p.view(p.shape[0], -1)
            reg.append(torch.norm(torch.mm(sigma, diff), 2)**2)
        loss += torch.sum(torch.stack(reg)) / n_samples

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(current_model.parameters(), 10)
    optimizer.step()
    
# Compute the quantile loss with anchored regularization for one network (parameter l2_reg)
def compute_td_loss_one_network(current_model,
                    target_model,
                    optimizer,
                    replay_buffer,
                    num_quant,
                    n_samples,
                    batch_size=64,
                    anchored=True):
    
    for i in range(100):
        if len(replay_buffer.buffer) < batch_size:
            X, action, reward = replay_buffer.sample_all()

        else:
            X, action, reward = replay_buffer.sample_batch(batch_size)


        X          = Variable(torch.FloatTensor(np.float32(X)))
        action     = Variable(torch.LongTensor(action))
        reward     = torch.FloatTensor(reward)

        train_step(current_model=current_model,
                   target_model=target_model,
                   optimizer=optimizer,
                  X=X,
                  action=action,
                  reward=reward,
                  n_samples=n_samples,
                  num_quant=num_quant,
                  anchored=anchored)
        
# Compute the quantile loss with anchored regularization for one network (parameter l2_reg)
def compute_td_loss_two_network(
                    current_model1,
                    target_model1,
                    optimizer1,
                    current_model2,
                    target_model2,
                    optimizer2,
                    replay_buffer,
                    num_quant,
                    n_samples,
                    batch_size=64,
                    anchored=True):
    
    for i in range(100):
        if len(replay_buffer.buffer) < batch_size:
            X, action, reward = replay_buffer.sample_all()

        else:
            X, action, reward = replay_buffer.sample_batch(batch_size)


        X          = Variable(torch.FloatTensor(np.float32(X)))
        action     = Variable(torch.LongTensor(action))
        reward     = torch.FloatTensor(reward)

        train_step(current_model=current_model1,
                   target_model=target_model1,
                   optimizer=optimizer1,
                   X=X,
                   action=action,
                   reward=reward,
                  n_samples=n_samples,
                  num_quant=num_quant,
                  anchored=anchored)

        train_step(current_model=current_model2,
                   target_model=target_model2,
                   optimizer=optimizer2,
                   X=X,
                   action=action,
                   reward=reward,
                  n_samples=n_samples,
                  num_quant=num_quant,
                  anchored=anchored)
    

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def play(data_reduced, batch_size, buffer_size, num_frames,
         current_model1, target_model1, optimizer1,
        current_model2=None, target_model2=None, optimizer2=None, anchored=True):
    
    # Set environment
    env = EnvMushroom(data_reduced)
    replay_buffer = ReplayBuffer(buffer_size)
    n_samples = 0
    
    all_reward = 0
    reward_list = []
    reward_cumul = []
    actions = []

    incertitudes = []

    all_regret = 0
    regret_cumul = []
    
    for frame_idx in tqdm(range(1, num_frames + 1)):
        X = env.sample()
        n_samples += 1
        if current_model2 is not None:
            action, incertitude = choose_action_Thompson(current_model1(X), current_model2(X), current_model1.num_quants)
            incertitudes.append(incertitude)
            action = action.item()
        
        elif current_model1.name == "QRDQN_dropout":
            action, incertitude = choose_action_Thompson(current_model1(X), current_model1(X), current_model1.num_quants)
            incertitudes.append(incertitude)
            action = action.item()

        
        else: # eps greedy
            action = current_model1.choose_action(X).item()
            
        actions.append(action)
        

        reward = env.hit(action)
        reward_list.append(reward)
        all_reward += reward
        reward_cumul.append(all_reward)

        all_regret += env.regret(action)
        regret_cumul.append(all_regret)

        replay_buffer.push(X, action, reward)
        if frame_idx % 10 == 0:
            if len(replay_buffer.buffer) > 2:
                if current_model2 is None:
                    compute_td_loss_one_network(current_model1,
                                    target_model1,
                                    optimizer1,
                                    replay_buffer,
                                    num_quant=current_model1.num_quants,
                                    n_samples = n_samples,
                                    batch_size = batch_size,
                                    anchored=anchored)

                else:
                    compute_td_loss_two_network(current_model1,
                                    target_model1,
                                    optimizer1,
                                    current_model2,
                                    target_model2,
                                    optimizer2,
                                    replay_buffer,
                                    num_quant=current_model1.num_quants,
                                    n_samples=n_samples,
                                    batch_size=batch_size,
                                    anchored=anchored)
        
        # Update target
        if frame_idx % (num_frames/10) == 0:
            update_target(current_model1, target_model1)
            if current_model2 is not None:
                update_target(current_model2, target_model2)
    
    return actions, incertitudes, reward_cumul, regret_cumul


# For an ensemble of networks :

def compute_ensemble_uncertainty(dists_stack):
    return dists_stack.std(0).mean(1).data * np.sqrt(len(dists_stack) / (len(dists_stack) - 1))

def choose_action_Thompson_ensemble(dists):
    dists_stack = torch.stack(dists).squeeze()
    
    uncertainty = compute_ensemble_uncertainty(dists_stack = torch.stack(dists).squeeze())
    
    mean_a = dists_stack.mean(0).mean(1)
    
    Q = [np.random.normal(loc=mean_a[k].item(),
                                   scale=i.item()) for k, i in enumerate(uncertainty.detach())]  
    
    Q = torch.tensor(Q)
    _, a_star = torch.max(Q, 0)

    return a_star, uncertainty.detach()

def choose_action_Thompson_ensemble_non_distributed(dists):
    dists_stack = torch.stack(dists)
    uncertainty = dists_stack.std(0)
    mean_a = dists_stack.mean(0)

    Q = [np.random.normal(loc=mean_a[k].item(),
                                   scale=i.item()) for k, i in enumerate(uncertainty.detach().numpy())]
    
    Q = torch.tensor(Q)
    _, a_star = torch.max(Q, 0)

    return a_star, uncertainty.detach()


def compute_td_loss_ensemble(
                    current_models,
                    target_models,
                    optimizers,
                    replay_buffer,
                    num_quant,
                    n_samples,
                    ensemble_size,
                    batch_size=64,
                    anchored=True):
    
    for i in range(100):
        if len(replay_buffer.buffer) < batch_size:
            X, action, reward = replay_buffer.sample_all()

        else:
            X, action, reward = replay_buffer.sample_batch(batch_size)


        X          = Variable(torch.FloatTensor(np.float32(X)))
        action     = Variable(torch.LongTensor(action))
        reward     = torch.FloatTensor(reward)

        for n in range(ensemble_size):
            train_step(current_model=current_models[n],
                       target_model=target_models[n],
                       optimizer=optimizers[n],
                       X=X,
                       action=action,
                       reward=reward,
                      n_samples=n_samples,
                      num_quant=num_quant,
                      anchored=anchored)
