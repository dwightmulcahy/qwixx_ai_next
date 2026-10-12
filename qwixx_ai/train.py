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
from .learning import potential, shaped_reward, terminal_reward

def train(episodes=2000,seed=42,out='runs',eval_games=80, target_interval=1000):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    obs_size=observe(Game()).size;policy=network(obs_size);target=network(obs_size)
    target.load_state_dict(policy.state_dict());opt=torch.optim.Adam(policy.parameters(),lr=5e-4)
    replay=deque(maxlen=50000);priorities=deque(maxlen=50000);snapshots=[];best=-1;updates=0
    log=folder/'metrics.jsonl'
    for ep in range(episodes):
        g=Game(seed=seed+ep);epsilon=max(.05,1-.95*ep/max(1,episodes*.95))
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
                    s,a,reward_start,start_potential=pending
                    base=(g.scores()[learner]-reward_start)/25
                    reward=shaped_reward(base,start_potential,potential(g,learner),False)
                    replay.append((s,a,reward,observe(g),False,mask(g)))
                    priorities.append(max(priorities,default=1.0))
                s=observe(g).copy();a=choose(policy,g,epsilon)
                pending=(s,encode(a),g.scores()[learner],potential(g,learner));g.step(a)
            else:
                pick=opponents[who]
                if pick=='heuristic':a=heuristic(g)
                elif pick=='random':a=random_agent(g)
                elif pick=='current':a=choose(policy,g,epsilon)
                else:a=choose(snapshots[pick],g)
                g.step(a)
            if len(replay)>=128:
                population=list(replay)
                weights=np.asarray(priorities,dtype=np.float64)**0.6
                probabilities=weights/weights.sum()
                indices=np.random.choice(len(population),size=64,replace=True,p=probabilities)
                batch=[population[int(i)] for i in indices]
                ss,aa,rr,ns,dd,mm=zip(*batch)
                q=policy(torch.tensor(np.stack(ss))).gather(1,torch.tensor(aa).long()[:,None]).squeeze(1)
                with torch.no_grad():
                    next_states=torch.from_numpy(np.stack(ns))
                    valid=torch.from_numpy(np.stack(mm))
                    online_next=policy(next_states).masked_fill(~valid,-1e9)
                    next_actions=online_next.argmax(1,keepdim=True)
                    next_values=target(next_states).gather(1,next_actions).squeeze(1)
                    expected=torch.tensor(rr,dtype=torch.float32)+.99*(1-torch.tensor(dd,dtype=torch.float32))*next_values
                td_errors=(q-expected).detach().abs().numpy()
                importance=(len(population)*probabilities[indices])**(-0.4)
                importance/=importance.max()
                individual_losses=nn.functional.smooth_l1_loss(q,expected,reduction='none')
                loss=(individual_losses*torch.tensor(importance,dtype=torch.float32)).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),5);opt.step();updates+=1
                for index,error in zip(indices,td_errors):
                    priorities[int(index)]=float(error)+0.01
                if updates%target_interval==0:target.load_state_dict(policy.state_dict())
        scores=g.scores()
        if pending is not None:
            s,a,start,start_potential=pending
            final=shaped_reward((scores[learner]-start)/25,start_potential,0.0,True)
            final+=terminal_reward(scores,learner)
            replay.append((s,a,final,observe(g),True,mask(g)))
            priorities.append(max(priorities,default=1.0))
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
    p.add_argument('--target-interval',type=int,default=1000)
    a=p.parse_args();train(a.episodes,a.seed,a.out,a.eval_games,a.target_interval)
if __name__=='__main__':main()
