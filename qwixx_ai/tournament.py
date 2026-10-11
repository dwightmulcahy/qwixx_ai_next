"""Reproducible round-robin seat rotation; split ties evenly."""
from .core import Game
from .agents import heuristic,random_agent
from .model import choose,load

def play(agents,seed):
    g=Game(n=len(agents),seed=seed)
    while not g.done:
        g.step(agents[g.actor](g))
    scores=g.scores();top=max(scores)
    winners=[i for i,s in enumerate(scores) if s==top]
    return scores,[1/len(winners) if i in winners else 0 for i in range(len(agents))]

def evaluate(model,games=100,opponent='heuristic',seed=1000):
    bot=lambda g:choose(model,g)
    other=heuristic if opponent=='heuristic' else random_agent
    wins=0;score=0
    for i in range(games):
        seat=i%4;agents=[other]*4;agents[seat]=bot
        scores,shares=play(agents,seed+i)
        wins+=shares[seat];score+=scores[seat]
    return {'games':games,'win_share':wins/games,'mean_score':score/games,'opponent':opponent}

def main():
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--games',type=int,default=200)
    p.add_argument('--opponent',choices=['heuristic','random'],default='heuristic');a=p.parse_args()
    print(json.dumps(evaluate(load(a.checkpoint),a.games,a.opponent),indent=2))
if __name__=='__main__':main()
