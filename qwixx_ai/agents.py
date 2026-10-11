import random
from .core import ORDER

def heuristic(game):
    moves=game.legal()
    if not moves:return None
    b=game.boards[game.actor]
    def value(a):
        c,n,lock=a;row=b.rows[c];index=ORDER[c].index(n)
        prev=ORDER[c].index(row[-1]) if row else -1
        skip=index-prev-1
        return (len(row)+1)*1.2-1.8*skip+7*lock
    best=max(value(a) for a in moves)
    # Passing is strategically sensible when available numbers are too far right.
    return max((a for a in moves if value(a)==best),key=lambda a:a[2]) if best>0 else None

def random_agent(game):
    return random.choice([None]+game.legal())
