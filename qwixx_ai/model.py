import random
import numpy as np
import torch
from torch import nn
from .env import ACTION_COUNT,observe,mask,decode

DUELING_ARCH="dueling-v1"
LEGACY_ARCH="mlp-v1"


class DuelingQNetwork(nn.Module):
    """Shared representation with separate state-value and action-advantage heads."""

    architecture=DUELING_ARCH

    def __init__(self,size):
        super().__init__()
        self.input_size=size
        self.trunk=nn.Sequential(
            nn.Linear(size,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
        )
        self.value=nn.Linear(128,1)
        self.advantage=nn.Linear(128,ACTION_COUNT)

    def forward(self,x):
        h=self.trunk(x)
        value=self.value(h)
        advantage=self.advantage(h)
        return value+advantage-advantage.mean(dim=-1,keepdim=True)

    def __getitem__(self,index):
        # Compatibility for older code/tests that inspect model[0].in_features.
        if index==0:return self.trunk[0]
        raise IndexError(index)


def legacy_network(size):
    model=nn.Sequential(
        nn.Linear(size,128),nn.ReLU(),
        nn.Linear(128,128),nn.ReLU(),
        nn.Linear(128,ACTION_COUNT),
    )
    model.architecture=LEGACY_ARCH
    model.input_size=size
    return model


def network(size):
    return DuelingQNetwork(size)


def model_architecture(model):
    return getattr(model,"architecture",LEGACY_ARCH)


def model_input_size(model):
    return getattr(model,"input_size",model[0].in_features)


def _linear_payload(layer):
    return {"w":layer.weight.detach().tolist(),"b":layer.bias.detach().tolist()}


def export_policy(model):
    """Serialize either the new dueling network or legacy sequential policy."""
    architecture=model_architecture(model)
    if architecture==DUELING_ARCH:
        trunk=[_linear_payload(layer) for layer in model.trunk if isinstance(layer,nn.Linear)]
        return {
            "architecture":DUELING_ARCH,
            "trunk":trunk,
            "value":_linear_payload(model.value),
            "advantage":_linear_payload(model.advantage),
        }
    layers=[_linear_payload(layer) for layer in model if isinstance(layer,nn.Linear)]
    return {"architecture":LEGACY_ARCH,"layers":layers}


def choose(model,game,epsilon=0):
    valid=mask(game)
    if random.random()<epsilon:return decode(int(random.choice(np.flatnonzero(valid))))
    with torch.no_grad():q=model(torch.from_numpy(observe(game))).numpy()
    q[~valid]=-1e9
    return decode(int(q.argmax()))


def save(model,path,**meta):
    torch.save({
        "model":model.state_dict(),
        "obs_size":observe(__import__("qwixx_ai.core",fromlist=["Game"]).Game()).size,
        "architecture":model_architecture(model),
        **meta,
    },path)


def load(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    architecture=checkpoint.get("architecture",LEGACY_ARCH)
    if architecture==DUELING_ARCH:
        model=network(checkpoint["obs_size"])
    elif architecture==LEGACY_ARCH:
        model=legacy_network(checkpoint["obs_size"])
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
