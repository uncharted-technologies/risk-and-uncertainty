#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import numpy as np
import torch


def set_global_seed(seed, env):
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def quantile_huber_loss(x,y, device, kappa=1):

    batch_size = x.shape[0] 
    num_quant = x.shape[1]

    #Get x and y to repeat here
    x = x.unsqueeze(2).repeat(1,1,num_quant)
    y = y.unsqueeze(2).repeat(1,1,num_quant).transpose(1,2)

    tau_hat = torch.linspace(0.0, 1.0 - 1. / num_quant, num_quant) + 0.5 / num_quant
    tau_hat = tau_hat.to(device)
    tau_hat = tau_hat.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,num_quant)
    
    diff = y-x

    if kappa == 0:
        huber_loss = diff.abs()
    else:
        huber_loss = 0.5 * diff.abs().clamp(min=0.0, max=kappa).pow(2)
        huber_loss += kappa * (diff.abs() - diff.abs().clamp(min=0.0, max=kappa))

    quantile_loss = (tau_hat - (diff < 0).float()).abs() * huber_loss

    return quantile_loss.mean(2).mean(0).sum()
    
