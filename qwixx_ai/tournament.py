"""Reproducible, seat-balanced evaluation utilities."""
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


def evaluation_seeds(games,seed=1000,seats=4):
    """Return matched seeds, one complete seat rotation per seed."""
    if games<seats or games%seats:
        raise ValueError(f'games must be a positive multiple of {seats}')
    rounds=games//seats
    return [seed+i for i in range(rounds)]


def evaluate(model,games=100,opponent='heuristic',seed=1000):
    """Evaluate a model with identical dice seeds in every seat.

    Every seed is replayed four times, once with the evaluated policy in each
    seat. This removes first-player/seat-order bias from promotion metrics.
    """
    bot=lambda g:choose(model,g)
    other=heuristic if opponent=='heuristic' else random_agent
    wins=0.0;score=0.0
    by_seat=[{'games':0,'win_share':0.0,'score':0.0} for _ in range(4)]
    seeds=evaluation_seeds(games,seed)
    for game_seed in seeds:
        for seat in range(4):
            agents=[other]*4;agents[seat]=bot
            scores,shares=play(agents,game_seed)
            wins+=shares[seat];score+=scores[seat]
            item=by_seat[seat]
            item['games']+=1;item['win_share']+=shares[seat];item['score']+=scores[seat]
    seat_results=[]
    for seat,item in enumerate(by_seat):
        n=item['games']
        seat_results.append({
            'seat':seat,'games':n,
            'win_share':item['win_share']/n,
            'mean_score':item['score']/n,
        })
    return {
        'games':games,'rounds':len(seeds),'win_share':wins/games,
        'mean_score':score/games,'opponent':opponent,'seed':seed,
        'seat_balanced':True,'by_seat':seat_results,
    }


def paired_compare(candidate,incumbent,games=192,opponent='heuristic',seed=1000):
    """Compare two models against the exact same seat/seed benchmark."""
    candidate_result=evaluate(candidate,games,opponent,seed)
    incumbent_result=evaluate(incumbent,games,opponent,seed)
    return {
        'games':games,'opponent':opponent,'seed':seed,'seat_balanced':True,
        'candidate':candidate_result,'incumbent':incumbent_result,
        'win_share_delta':candidate_result['win_share']-incumbent_result['win_share'],
        'mean_score_delta':candidate_result['mean_score']-incumbent_result['mean_score'],
    }


def main():
    import argparse,json,random
    import numpy as np
    import torch
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--games',type=int,default=200)
    p.add_argument('--opponent',choices=['heuristic','random','both'],default='heuristic')
    p.add_argument('--seed',type=int,default=1000)
    a=p.parse_args()
    model=load(a.checkpoint)
    results=[]
    for opponent in (['heuristic','random'] if a.opponent=='both' else [a.opponent]):
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        results.append(evaluate(model,a.games,opponent,seed=a.seed))
    print(json.dumps({'checkpoint':a.checkpoint,'seed':a.seed,'results':results},indent=2))


if __name__=='__main__':main()
