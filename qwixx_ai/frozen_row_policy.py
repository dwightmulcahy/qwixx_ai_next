"""Frozen row-only model, independent of the champion entrypoint."""
from .decision_quality import features

WEIGHTS = (
    7.316526290373287e-34, 0.0, 9.190012946650607e-34, 0.0, 0.0,
    -0.020710467226715008, 0.05338450113214413,
    -0.0926056063043526, -0.009760136736374748,
    -0.005655617225870696, 0.046577911421847515,
    0.08235247612364781, 0.046211322675620656,
    0.027243457944153966,
)
MARGIN = 0.02


def model_choice(game, incumbent):
    actions = [None, *game.legal()]
    values = {
        action: sum(weight * value for weight, value in zip(
            WEIGHTS, features(game, action)))
        for action in actions
    }
    best = max(actions, key=lambda action: values[action])
    return best if values[best] > values[incumbent] + MARGIN else incumbent


def row_change_only_from_choices(incumbent, proposed):
    if (incumbent is not None and proposed is not None
            and not incumbent[2] and not proposed[2]
            and incumbent[0] != proposed[0]):
        return proposed
    return incumbent


def row_change_only(game, incumbent):
    return row_change_only_from_choices(incumbent, model_choice(game, incumbent))
