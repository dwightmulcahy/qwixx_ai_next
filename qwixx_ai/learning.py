"""Shared learning rewards and outcome accounting."""

def terminal_reward(scores, seat):
    """Fractional win credit for ties; -1 for non-winning seats."""
    winners=sum(score==max(scores) for score in scores)
    return 1.0/winners if scores[seat]==max(scores) else -1.0
