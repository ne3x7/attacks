import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
from copy import deepcopy

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
def compute_bounds(model, x, eps=0.1):
    """
    Computes lower and upper bounds for z_k as equation (6) in 
    
        https://arxiv.org/abs/1810.12715
        
    """
    bounds = []
    
    # if we have convolutional layers we need to reshape input to flat vectors before FC layers 
    
    bounds.append((x - eps, x + eps))
    
    for layer in model.layers:
        
        z_l_prev, z_u_prev = bounds[-1]
        
        if isinstance(layer, Flatten):
            z_l = Flatten()(z_l_prev)
            z_u = Flatten()(z_u_prev)
            
            bounds.append((z_l, z_u))
        
        if isinstance(layer, nn.Linear):
            W = layer.weight
            b = layer.bias

            m = (z_u_prev + z_l_prev) / 2
            r = (z_u_prev - z_l_prev) / 2

            m = torch.matmul(m, W.t()) + b
            r = torch.matmul(r, torch.abs(W.t()))

            z_l = m - r
            z_u = m + r
            bounds.append((z_l, z_u))
            
        elif isinstance(layer, nn.Conv2d):
            z_l = (F.conv2d(z_l_prev, layer.weight.clamp(min=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  F.conv2d(z_u_prev, layer.weight.clamp(max=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None,:,None,None])
            
            z_u = (F.conv2d(z_u_prev, layer.weight.clamp(min=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  F.conv2d(z_l_prev, layer.weight.clamp(max=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) + 
                  layer.bias[None,:,None,None])
            bounds.append((z_l, z_u))

        elif isinstance(layer, nn.ReLU):
            
            z_l = torch.relu(z_l_prev)
            z_u = torch.relu(z_u_prev)
            
            bounds.append((z_l, z_u))
        
        else: 
            continue
        
    return bounds

def ibp_loss(base_loss, logits, z_k, y, k=0.5):

    e_y = torch.zeros((1, logits.shape[1]))
    e_y[:, y] = 1
    z_hat =  e_y * z_k[0] + (1 - e_y) * z_k[1]
    
    L_spec = base_loss(z_hat, y)
    L_fit = base_loss(logits, y)
    
    L = k * L_fit + (1 - k) * L_spec
    
    return L, L_spec, L_fit

def curvature_regularization_loss(model, lossf, x, h, y=None, attack_norm_p='inf'):
    """
    Computes curvature regularization term.
    
    The formula is L(x) = \| \nabla l(x + h z) - \nabla l(x) \|^2,
    where z depends on attack norm. If attack is in \ell_inf, then
    z = sign \nabla l(x) / \| sign \nabla l(x) \|. Another good
    choice is z = \nabla l(x) / \| \nabla l(x) \|.
    
    Args:
        model, lossf (Module): model and corresponding loss function
        x, y (Tensor): data and optional label
        h (float): interpolation parameter
        attack_norm_p (str): if 'inf', \ell_inf z is used, otherwise
            simply normalized gradient.
    """
    original = x.clone().detach().requires_grad_(True)
    prob_original = lossf(model(original), y) if y is not None else lossf(model(original))
    gradients_original = torch.autograd.grad(outputs=prob_original,
                                             inputs=original,
                                             grad_outputs=torch.ones(prob_original.size()),
                                             create_graph=True,
                                             retain_graph=True)[0]
    
    # do not back-propagate through z
    if attack_norm_p == 'inf':
        z = gradients_original.clone().detach().sign()
    else:
        z = gradients_original.clone().detach()
    
    interpolated = (x + h * z).requires_grad_(True)
    prob_interpolated = lossf(model(interpolated), y) if y is not None else lossf(model(interpolated))
    gradients_interpolated = torch.autograd.grad(outputs=prob_interpolated,
                                                 inputs=interpolated,
                                                 grad_outputs=torch.ones(prob_interpolated.size()),
                                                 create_graph=True,
                                                 retain_graph=True)[0]

    return torch.sum((gradients_interpolated - gradients_original) ** 2)
        
class Net(nn.Module):
    def __init__(self, n_channels=1, fc_dim=676):
        super().__init__()
        
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1, stride=2)
        
        self.flat = Flatten()
        
        self.fc = nn.Linear(int(20*fc_dim/4), 10)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        
        self.layers = nn.Sequential(
            self.conv1,
            self.act1,
            self.conv2,
            self.act2,
            self.flat,
            self.fc
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 64, kernel_size=3)
        
        self.flat = Flatten()
        
        self.fc = nn.Linear(43264, 10)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        
        self.layers = nn.Sequential(
            self.conv1,
            self.act1,
            self.conv2,
            self.act2,
            self.flat,
            self.fc
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x