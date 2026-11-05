"""Dueling Q-network architecture regressions."""
import torch

from qwixx_ai.env import ACTION_COUNT
from qwixx_ai.model import (
    DUELING_ARCH,
    export_policy,
    legacy_network,
    load,
    model_architecture,
    network,
    save,
)


def test_dueling_network_output_shape_and_zero_mean_advantage_composition():
    model=network(262)
    x=torch.zeros(262)
    q=model(x)
    assert q.shape==(ACTION_COUNT,)
    assert model_architecture(model)==DUELING_ARCH


def test_dueling_policy_export_has_separate_value_and_advantage_heads():
    payload=export_policy(network(262))
    assert payload["architecture"]==DUELING_ARCH
    assert len(payload["trunk"])==3
    assert len(payload["value"]["b"])==1
    assert len(payload["advantage"]["b"])==ACTION_COUNT


def test_save_load_preserves_dueling_architecture(tmp_path):
    path=tmp_path/"dueling.pt"
    model=network(262)
    save(model,path)
    restored=load(path)
    assert model_architecture(restored)==DUELING_ARCH
    assert restored(torch.zeros(262)).shape==(ACTION_COUNT,)


def test_legacy_checkpoint_loader_remains_available_for_research(tmp_path):
    path=tmp_path/"legacy.pt"
    old=legacy_network(262)
    torch.save({"model":old.state_dict(),"obs_size":262},path)
    restored=load(path)
    assert model_architecture(restored)=="mlp-v1"
    assert restored(torch.zeros(262)).shape==(ACTION_COUNT,)
