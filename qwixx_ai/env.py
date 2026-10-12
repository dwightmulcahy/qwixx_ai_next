"""Gymnasium-compatible environment. Gymnasium is optional for core tests/web."""
import numpy as np
from .core import Game,ORDER
from .agents import heuristic,random_agent
from .learning import terminal_reward, potential, shaped_reward
try:
    import gymnasium as gym
    from gymnasium import spaces
    Base=gym.Env
except ImportError:
    gym=None;spaces=None;Base=object
ACTION_COUNT=89

def encode(action):
    if action is None:return 0
    c,n,lock=action
    return 1+22*c+2*ORDER[c].index(n)+int(lock)

def decode(a):
    if a==0:return None
    if not 0<a<ACTION_COUNT:raise ValueError(a)
    x=a-1;c=x//22
    return (c,ORDER[c][(x%22)//2],bool(x%2))

def observe(game):
    v=[]
    # All public boards, not only our own. Actor and phase explicitly included.
    for b in game.boards:
        for c in range(4):v.extend(float(n in b.rows[c]) for n in ORDER[c])
        v.append(b.penalties/4)
    # Strategic features for each seat/row: progress, remaining opportunities,
    # next-mark value, and lock readiness. All features remain in [0, 1].
    for b in game.boards:
        for c, order in enumerate(ORDER):
            row=b.rows[c]
            last=order.index(row[-1]) if row else -1
            remaining=0 if c in game.locks else 10-last
            v.extend((len(row)/11, remaining/11,
                      min(len(row)+1, 12)/12,
                      float(len(row)>=5 and c not in game.locks)))
        v.append(b.penalties/4)
    v.extend(float(c in game.locks) for c in range(4))
    v.extend(d/6 for d in game.dice)
    v.extend([game.active/game.n,game.actor/game.n,game.phase/(game.n+1),min(game.turn/100,1),game.active_marks/2])
    return np.array(v,dtype=np.float32)

def mask(game):
    a=np.zeros(ACTION_COUNT,dtype=np.bool_);a[0]=True
    for x in game.legal():a[encode(x)]=True
    return a

class QwixxEnv(Base):
    metadata={'render_modes':['ansi']}
    def __init__(self,opponent='heuristic',seed=None):
        self.game=Game(seed=seed);self.opponent=opponent
        self._seed_rng=np.random.default_rng(seed)
        if spaces:
            self.action_space=spaces.Discrete(ACTION_COUNT)
            self.observation_space=spaces.Box(0,1,shape=observe(self.game).shape,dtype=np.float32)
    def action_masks(self):return mask(self.game)
    def _advance(self):
        while not self.game.done and self.game.actor!=0:
            a=random_agent(self.game) if self.opponent=='random' else heuristic(self.game)
            self.game.step(a)
    def reset(self,seed=None,options=None):
        if gym:super().reset(seed=seed)
        if seed is not None:self._seed_rng=np.random.default_rng(seed)
        self.game=Game(seed=int(self._seed_rng.integers(0,2**32)));self._advance()
        return observe(self.game),{'action_mask':mask(self.game)}
    def step(self,action):
        if self.game.done:raise RuntimeError('Game finished')
        if not mask(self.game)[int(action)]:raise ValueError('Illegal action')
        before=self.game.scores()[0]
        previous_potential=potential(self.game,0)
        self.game.step(decode(int(action)));self._advance()
        scores=self.game.scores();done=self.game.done
        reward=shaped_reward((scores[0]-before)/25,previous_potential,
                             potential(self.game,0),done)
        if done:reward+=terminal_reward(scores,0)
        return observe(self.game),float(reward),done,False,{'action_mask':mask(self.game),'scores':scores}
