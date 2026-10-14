"""Shared-policy multi-agent self-play DQN with historical snapshot opponents."""
import argparse,random,json
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import Game
from .env import observe,mask,encode
from .model import network,choose,save,load
from .agents import heuristic,random_agent
from .tournament import evaluate
from .learning import potential, shaped_reward, terminal_reward
from .league import freeze, promote
from .phase5 import evaluate_promotion
from .phase52 import evaluate_promotion as evaluate_promotion_v52
from .phase53 import evaluate_promotion as evaluate_promotion_v53
from .phase54 import select_checkpoint, exploration_rate, opponent_pool

def train(episodes=2000,seed=42,out='runs',eval_games=80, target_interval=1000, champion_path=None, promotion_games=1000, warm_start=False, phase5=False, historical_paths=(), learning_rate=None,
          eval_interval=None, rollback_patience=3, rollback_floor=0.22,
          phase52=False, promotion_alpha=0.05, phase53=False, phase54=False,
          phase55=False, anchor_weight=0.05, update_every=4,
          init_checkpoint=None, terminal_win_weight=1.0, relative_score_weight=0.0):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if relative_score_weight < 0:
        raise ValueError('relative_score_weight must be nonnegative')
    if terminal_win_weight < 0:
        raise ValueError('terminal_win_weight must be nonnegative')
    if init_checkpoint is not None and (champion_path is not None or warm_start):
        raise ValueError('init_checkpoint is separate from champion/warm_start')
    if episodes < 1 or eval_games < 1 or target_interval < 1:
        raise ValueError('episodes, eval_games and target_interval must be positive')
    if rollback_patience < 1 or not 0 <= rollback_floor <= 1:
        raise ValueError('invalid rollback settings')
    interval=eval_interval or max(1,episodes//10)
    if interval < 1:
        raise ValueError('eval_interval must be positive')
    lr=learning_rate if learning_rate is not None else (1e-4 if warm_start else 5e-4)
    if lr <= 0:
        raise ValueError('learning_rate must be positive')
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    obs_size=observe(Game()).size;policy=network(obs_size);target=network(obs_size)
    if init_checkpoint is not None:
        initial=load(init_checkpoint)
        if initial[0].in_features != obs_size:
            raise ValueError('Initial checkpoint has incompatible observations')
        policy.load_state_dict(initial.state_dict())
    target.load_state_dict(policy.state_dict());opt=torch.optim.Adam(policy.parameters(),lr=lr)
    replay=deque(maxlen=50000);priorities=deque(maxlen=50000);snapshots=[];best=-1;updates=0
    log=folder/'metrics.jsonl'
    champion=None
    if champion_path is not None:
        champion=freeze(load(champion_path))
        if champion[0].in_features != obs_size:
            raise ValueError('Champion checkpoint has incompatible observations')
        snapshots.append(champion)
        if warm_start:
            policy.load_state_dict(champion.state_dict())
            target.load_state_dict(policy.state_dict())
    elif warm_start:
        raise ValueError('--warm-start requires --champion')
    historical=[freeze(load(path)) for path in historical_paths]
    original_champion=freeze(champion) if champion is not None else None
    permanent_opponents=([original_champion]+historical if champion is not None else [])
    if any(model[0].in_features != obs_size for model in historical):
        raise ValueError('Historical checkpoint has incompatible observations')
    weak_evals=0
    best_competitive=-1.0
    best_league_score=float('-inf')
    best_candidate_score=float('-inf')
    game_steps=0
    anchor=freeze(champion) if phase55 and champion is not None else None
    if phase55 and not phase54:
        raise ValueError('--phase55 requires --phase54')
    if phase55 and (anchor_weight < 0 or update_every < 1):
        raise ValueError('invalid Phase 5.5 regularization or update interval')
    if phase54 and not phase53:
        raise ValueError('--phase54 requires --phase53')
    if phase54 and len(historical_paths) < 2:
        raise ValueError('--phase54 requires at least two historical checkpoints')
    if phase53 and not phase52:
        raise ValueError('--phase53 requires --phase52')
    if phase52 and not phase5:
        raise ValueError('--phase52 requires --phase5')
    if phase5 and champion is None:
        raise ValueError('--phase5 requires --champion')
    for ep in range(episodes):
        g=Game(seed=seed+ep)
        epsilon=(exploration_rate(ep, episodes) if phase54 else
                 max(.05,.20-.15*ep/max(1,episodes*.9)) if (warm_start or init_checkpoint is not None)
                 else max(.05,1-.95*ep/max(1,episodes*.95)))
        # One learner seat, rotated; opponents from current policy, frozen snapshots or baselines.
        learner=ep%4;opponents=[]
        for seat in range(4):
            if seat==learner:opponents.append(None)
            else:
                if champion is None:
                    candidates=['current','heuristic','random']+list(range(len(snapshots)))
                elif phase54:
                    candidates=opponent_pool(len(permanent_opponents), len(snapshots))
                else:
                    progress=ep/max(1,episodes-1)
                    champion_weight=2+int(4*progress)
                    candidates=['current','heuristic','random']+[0]*champion_weight
                    candidates+=list(range(1,len(snapshots)))*2
                    candidates += [('permanent',i) for i in range(len(permanent_opponents))]*2
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
                elif isinstance(pick,tuple):a=choose(permanent_opponents[pick[1]],g)
                else:a=choose(snapshots[pick],g)
                g.step(a)
            game_steps+=1
            if len(replay)>=128 and (not phase55 or game_steps % update_every == 0):
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
                if phase55:
                    with torch.no_grad():
                        anchor_q=anchor(torch.tensor(np.stack(ss)))
                    anchor_actions=anchor_q.argmax(dim=1,keepdim=True)
                    anchor_values=anchor_q.gather(1,anchor_actions).squeeze(1)
                    policy_values=policy(torch.tensor(np.stack(ss))).gather(1,anchor_actions).squeeze(1)
                    loss=loss+anchor_weight*nn.functional.smooth_l1_loss(policy_values,anchor_values)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),5);opt.step();updates+=1
                for index,error in zip(indices,td_errors):
                    priorities[int(index)]=float(error)+0.01
                if updates%target_interval==0:target.load_state_dict(policy.state_dict())
        scores=g.scores()
        if pending is not None:
            s,a,start,start_potential=pending
            final=shaped_reward((scores[learner]-start)/25,start_potential,0.0,True)
            final+=terminal_win_weight*terminal_reward(scores,learner)
            opponent_best=max(score for i,score in enumerate(scores) if i!=learner)
            final+=relative_score_weight*max(-1.0,min(1.0,
                (scores[learner]-opponent_best)/25.0))
            replay.append((s,a,final,observe(g),True,mask(g)))
            priorities.append(max(priorities,default=1.0))
        if (ep+1)%interval==0 or ep==episodes-1:
            policy.eval();result=evaluate(policy,eval_games,seed=100000)
            row={'episode':ep+1,'epsilon':round(epsilon,4),
                 'terminal_win_weight':terminal_win_weight,
                 'relative_score_weight':relative_score_weight,
                 'init_checkpoint':str(init_checkpoint) if init_checkpoint else None,
                 **result}
            with log.open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if result['win_share']>best:
                best=result['win_share'];save(policy,folder/'best.pt',episode=ep+1,win_share=best)
            save(policy,folder/'latest.pt',episode=ep+1)
            if champion is not None:
                if phase54:
                    duel=select_checkpoint(
                        policy, champion, historical,
                        rounds=max(2,promotion_games//8),
                        seed=900000+ep+1,
                        alpha=promotion_alpha,
                        max_checks=(episodes+interval-1)//interval,
                        lineup=permanent_opponents[:3],
                        incumbent_score=best_league_score)
                    if phase55 and duel['selection']['score'] > best_candidate_score:
                        best_candidate_score=duel['selection']['score']
                        save(policy,folder/'best_candidate.pt',episode=ep+1,
                             league_score=best_candidate_score)
                    if duel['selection']['selected']:
                        best_league_score=duel['selection']['score']
                        save(policy,folder/'best_competitive.pt',episode=ep+1,
                             league_score=best_league_score)
                elif phase53:
                    duel=evaluate_promotion_v53(
                        policy,champion,historical,
                        rounds=max(2,promotion_games//8),seed=900000+ep+1,
                        alpha=promotion_alpha,
                        max_checks=(episodes+interval-1)//interval,
                        lineup=permanent_opponents[:3] if len(permanent_opponents)>=3 else None,
                        destination=folder/'champion.pt')
                elif phase52:
                    duel=evaluate_promotion_v52(
                        policy,champion,historical,
                        rounds=max(2,promotion_games//8),seed=900000+ep+1,
                        alpha=promotion_alpha,
                        max_checks=(episodes+interval-1)//interval,
                        destination=folder/'champion.pt')
                elif phase5:
                    duel=evaluate_promotion(
                        policy,champion,historical,
                        rounds=max(2,promotion_games//4),seed=900000+ep+1,
                        destination=folder/'champion.pt')
                else:
                    duel=promote(policy,champion,folder/'champion.pt',games=promotion_games,
                                 seed=900000+ep+1)
                row['head_to_head']=duel
                if phase5:
                    primary=duel['primary']
                    competitive_score=(primary['paired_advantage'] if phase52
                                       else primary['win_share'])
                    history_guard=(not phase53 or all(
                        item['paired_advantage'] >= 0
                        for item in duel['historical']))
                    if not phase54 and history_guard and competitive_score > best_competitive:
                        best_competitive=competitive_score
                        save(policy,folder/'best_competitive.pt',episode=ep+1,
                             win_share=best_competitive)
                    weak=(primary['advantage_ci_high'] < rollback_floor-0.25
                          if phase52 else primary['ci95_high'] < rollback_floor)
                    if weak:
                        weak_evals+=1
                    else:
                        weak_evals=0
                    if weak_evals >= rollback_patience:
                        policy.load_state_dict(champion.state_dict())
                        target.load_state_dict(policy.state_dict())
                        opt=torch.optim.Adam(policy.parameters(),lr=lr)
                        replay.clear();priorities.clear();updates=0
                        weak_evals=0
                        rollback={'episode':ep+1,'rollback':True,
                                  'reason':'confidence upper bound below rollback floor',
                                  'floor':rollback_floor}
                        with log.open('a') as f:f.write(json.dumps(rollback)+'\n')
                        print(json.dumps(rollback),flush=True)
                if duel['promoted']:
                    if phase54:
                        save(policy,folder/'champion.pt',episode=ep+1,
                             league_score=duel['selection']['score'])
                    if phase52:
                        archive=folder/'champions'
                        archive.mkdir(parents=True,exist_ok=True)
                        save(policy,archive/f'episode_{ep+1:07d}.pt',
                             episode=ep+1,paired_advantage=duel['primary']['paired_advantage'])
                    champion=freeze(policy)
                    snapshots[0]=champion
                    weak_evals=0
                with log.open('a') as f:f.write(json.dumps({'episode':ep+1,'head_to_head':duel})+'\n')
                print(json.dumps({'episode':ep+1,'head_to_head':duel}),flush=True)
            if len(snapshots)<8:
                snapshot=network(obs_size);snapshot.load_state_dict(policy.state_dict());snapshot.eval();snapshots.append(snapshot)
            else:
                idx=((ep+1)//max(1,episodes//10))%7+1 if champion is not None else ((ep+1)//max(1,episodes//10))%8
                snapshots[idx].load_state_dict(policy.state_dict());snapshots[idx].eval()
            policy.train()
    return best

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--out',default='runs')
    p.add_argument('--eval-games',type=int,default=80)
    p.add_argument('--target-interval',type=int,default=1000)
    p.add_argument('--champion',default=None,help='Existing champion checkpoint for league training')
    p.add_argument('--promotion-games',type=int,default=1000)
    p.add_argument('--warm-start',action='store_true',help='Initialize policy and target from champion')
    p.add_argument('--phase5',action='store_true',help='Use paired-seed confidence-gated promotion')
    p.add_argument('--historical',nargs='*',default=[],help='Historical champion checkpoints')
    p.add_argument('--learning-rate',type=float,default=None)
    p.add_argument('--eval-interval',type=int,default=None)
    p.add_argument('--rollback-patience',type=int,default=3)
    p.add_argument('--rollback-floor',type=float,default=0.22)
    p.add_argument('--phase52',action='store_true',help='Use symmetric paired promotion and archive champions')
    p.add_argument('--promotion-alpha',type=float,default=0.05)
    p.add_argument('--phase53',action='store_true',help='Controlled fixed-lineup paired evaluation')
    p.add_argument('--phase54',action='store_true',help='Lower exploration, stronger league and multi-opponent selection')
    p.add_argument('--phase55',action='store_true',help='Conservative updates, champion anchor, and unconditional best candidate')
    p.add_argument('--anchor-weight',type=float,default=0.05)
    p.add_argument('--update-every',type=int,default=4)
    a=p.parse_args();train(a.episodes,a.seed,a.out,a.eval_games,a.target_interval,
                           a.champion,a.promotion_games,a.warm_start,a.phase5,a.historical,
                           a.learning_rate,a.eval_interval,a.rollback_patience,
                           a.rollback_floor,a.phase52,a.promotion_alpha,a.phase53,a.phase54,
                           a.phase55,a.anchor_weight,a.update_every)
if __name__=='__main__':main()
