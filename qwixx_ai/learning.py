"""Shared learning rewards and outcome accounting."""

def terminal_reward(scores, seat):
    """Fractional win credit for ties; -1 for non-winning seats."""
    top=max(scores)
    winners=sum(score==top for score in scores)
    return 1.0/winners if scores[seat]==top else -1.0
