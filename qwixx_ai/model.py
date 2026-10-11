import random
import numpy as np
import torch
from torch import nn
from .env import ACTION_COUNT,observe,mask,decode

def network(size):
    return nn.Sequential(nn.Linear(size,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,ACTION_COUNT))

def choose(model,game,epsilon=0):
    valid=mask(game)
    if random.random()<epsilon:return decode(int(random.choice(np.flatnonzero(valid))))
    with torch.no_grad():q=model(torch.from_numpy(observe(game))).numpy()
    q[~valid]=-1e9
    return decode(int(q.argmax()))

def save(model,path,**meta):
    torch.save({'model':model.state_dict(),'obs_size':observe(__import__('qwixx_ai.core',fromlist=['Game']).Game()).size,**meta},path)

def load(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    m=network(checkpoint['obs_size']);m.load_state_dict(checkpoint['model']);m.eval();return m
