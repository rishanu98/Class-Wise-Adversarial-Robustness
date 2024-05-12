from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt
#from utils.AverageMeter import AverageMeter
#from utils.criterion import *
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing as preprocessing
import sys
from math import pi
from math import cos
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                      
def pgd_linf(model, X, y, epsilon=0.031, alpha=0.01, num_iter=10, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_targ2(self, x_natural, y_targ, epis, alp, k):
        'Construct PGD targeted examples on example X'
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epis, epis)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                targeted_labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device).fill_(y_targ[0])
                loss = F.cross_entropy(logits[:, y_targ], targeted_labels)
       
            grad = torch.autograd.grad(loss, [x])[0]
          
            x = x.detach() + alp * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epis), x_natural + epis)
            x = torch.clamp(x, 0, 1)

        return x

def mixup_data(x, y, alpha = 0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha).cpu()
    else:
        lam = np.random.beta(alpha, alpha).cpu()

    batch_size = x.size()[0]

    if device == 'cuda':
        index = torch.randperm(batch_size).cuda() # generates a random permutation of indices for shuffling the samples. This is used to select a random sample from the dataset to mix with the current sample.
    else:
        index = torch.randperm(batch_size)
        
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y
    