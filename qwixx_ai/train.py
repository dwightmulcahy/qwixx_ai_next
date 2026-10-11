"""Shared-policy multi-agent self-play DQN with historical snapshot opponents."""
import argparse,random,json
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import Game
from .env import observe,mask,encode
from .model import network,choose,save
from .agents import heuristic,random_agent
from .tournament import evaluate

def train(episodes=2000,seed=42,out='runs',eval_games=80):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    obs_size=observe(Game()).size;policy=network(obs_size);target=network(obs_size)
    target.load_state_dict(policy.state_dict());opt=torch.optim.Adam(policy.parameters(),lr=5e-4)
    replay=deque(maxlen=50000);snapshots=[];best=-1;updates=0
    log=folder/'metrics.jsonl'
    for ep in range(episodes):
        g=Game(seed=seed+ep);epsilon=max(.05,1-.95*ep/max(1,episodes*.8))
        # One learner seat, rotated; opponents from current policy, frozen snapshots or baselines.
        learner=ep%4;opponents=[]
        for seat in range(4):
            if seat==learner:opponents.append(None)
            else:
                candidates=['current','heuristic','random']+list(range(len(snapshots)))
                pick=random.choice(candidates)
                opponents.append(pick)
        pending=None
        while not g.done:
            who=g.actor
            if who==learner:
                if pending is not None:
                    s,a,reward_start=pending
                    replay.append((s,a,(g.scores()[learner]-reward_start)/25,observe(g),False,mask(g)))
                s=observe(g).copy();a=choose(policy,g,epsilon)
                pending=(s,encode(a),g.scores()[learner]);g.step(a)
            else:
                pick=opponents[who]
                if pick=='heuristic':a=heuristic(g)
                elif pick=='random':a=random_agent(g)
                elif pick=='current':a=choose(policy,g,epsilon)
                else:a=choose(snapshots[pick],g)
                g.step(a)
            if len(replay)>=128:
                batch=random.sample(replay,64)
                ss,aa,rr,ns,dd,mm=zip(*batch)
                q=policy(torch.tensor(np.stack(ss))).gather(1,torch.tensor(aa).long()[:,None]).squeeze(1)
                with torch.no_grad():
                    future=target(torch.tensor(np.stack(ns)))
                    future[~torch.tensor(np.stack(mm))]=-1e9
                    expected=torch.tensor(rr,dtype=torch.float32)+.99*(1-torch.tensor(dd,dtype=torch.float32))*future.max(1).values
                loss=nn.functional.smooth_l1_loss(q,expected)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),5);opt.step();updates+=1
                if updates%500==0:target.load_state_dict(policy.state_dict())
        scores=g.scores()
        if pending is not None:
            s,a,start=pending
            final=(scores[learner]-start)/25+(1 if scores[learner]==max(scores) else -1)
            replay.append((s,a,final,observe(g),True,mask(g)))
        if (ep+1)%max(1,episodes//10)==0 or ep==episodes-1:
            policy.eval();result=evaluate(policy,eval_games,seed=100000)
            row={'episode':ep+1,'epsilon':round(epsilon,4),**result}
            with log.open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if result['win_share']>best:
                best=result['win_share'];save(policy,folder/'best.pt',episode=ep+1,win_share=best)
            save(policy,folder/'latest.pt',episode=ep+1)
            if len(snapshots)<8:
                snapshot=network(obs_size);snapshot.load_state_dict(policy.state_dict());snapshot.eval();snapshots.append(snapshot)
            else:
                idx=((ep+1)//max(1,episodes//10))%8
                snapshots[idx].load_state_dict(policy.state_dict());snapshots[idx].eval()
            policy.train()
    return best

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--out',default='runs')
    p.add_argument('--eval-games',type=int,default=80)
    a=p.parse_args();train(a.episodes,a.seed,a.out,a.eval_games)
if __name__=='__main__':main()
