"""Phase 7.2 hybrid policies: explicit, auditable action-selection rules."""
from .core import ORDER
from .heuristic_v2 import heuristic_v2
from .model import choose


def large_jump(game, action, threshold=3):
    if action is None:
        return False
    color, number, _ = action
    row = game.boards[game.actor].rows[color]
    previous = ORDER[color].index(row[-1]) if row else -1
    return ORDER[color].index(number) - previous - 1 >= threshold


def select(game, model, mode="student"):
    """Select among legal student and V2 decisions without lookahead."""
    if mode not in {"student", "v2", "pass_correction",
                    "large_jump", "selective"}:
        raise ValueError("unknown hybrid mode")
    if mode == "v2":
        return heuristic_v2(game)
    student = choose(model, game)
    if mode == "student":
        return student
    teacher = heuristic_v2(game)
    if student == teacher:
        return student
    if mode == "pass_correction":
        return teacher if student is None or teacher is None else student
    if mode == "large_jump":
        return student if large_jump(game, student) else teacher
    # Selective: correct pass disagreements, retain student large jumps,
    # and otherwise use V2. This is a hypothesis, not a validated improvement.
    if student is None or teacher is None:
        return teacher
    return student if large_jump(game, student) else teacher


def policy(model, mode):
    def act(game):
        return select(game, model, mode)
    return act
