"""Standard Qwixx with simultaneous white-dice decisions resolved in seat order."""
from dataclasses import dataclass, field
import random
COLORS=('red','yellow','green','blue')
ORDER=tuple(tuple(range(2,13)) if i<2 else tuple(range(12,1,-1)) for i in range(4))
# Action: None=pass, or (color index, number, lock flag).
@dataclass
class Board:
    rows:list=field(default_factory=lambda:[[] for _ in range(4)])
    penalties:int=0
    def can(self,c,n,locks):
        return (c not in locks and n in ORDER[c]
                and (not self.rows[c] or ORDER[c].index(n)>ORDER[c].index(self.rows[c][-1]))
                and (n != ORDER[c][-1] or len(self.rows[c]) >= 5))
    def can_lock(self,c,n,locks):
        return self.can(c,n,locks) and n==ORDER[c][-1] and len(self.rows[c])>=5
    def score(self,locks):
        total=0
        for c,row in enumerate(self.rows):
            k=len(row)+(1 if c in locks and row and row[-1]==ORDER[c][-1] else 0)
            total+=k*(k+1)//2
        return total-5*self.penalties

class Game:
    def __init__(self,n=4,seed=None):
        if not 2<=n<=5:raise ValueError('2–5 players supported')
        self.n=n;self.rng=random.Random(seed);self.reset()
    def reset(self):
        self.boards=[Board() for _ in range(self.n)];self.locks=set();self.active=0
        self.phase=0;self.turn=0;self.active_marks=0;self.white_locks=set();self._roll();return self
    def _roll(self):
        self.dice=tuple(self.rng.randint(1,6) for _ in range(6));self.phase=0;self.active_marks=0;self.white_locks=set()
    @property
    def actor(self):
        if self.phase==0 or self.phase==self.n:return self.active
        return (self.active+self.phase)%self.n
    @property
    def done(self):return (len(self.locks)>=2 and not (self.phase<self.n and self.white_locks)) or any(b.penalties>=4 for b in self.boards)
    @property
    def available_colors(self):
        """Colored dice still in play; locked dice cannot be selected."""
        return tuple(c for c in range(4) if c not in self.locks)
    def scores(self):return [b.score(self.locks) for b in self.boards]
    def legal(self):
        b=self.boards[self.actor]
        # White-dice decisions are simultaneous: a row locked earlier in this
        # phase remains selectable by players who have not yet acted.
        effective_locks=self.locks-self.white_locks if self.phase<self.n else self.locks
        if self.phase==self.n:
            candidates={(c,self.dice[w]+self.dice[c+2]) for c in range(4) if c not in effective_locks for w in (0,1)}
        else:
            candidates={(c,self.dice[0]+self.dice[1]) for c in range(4) if c not in effective_locks}
        out=[]
        for c,num in sorted(candidates):
            if b.can(c,num,effective_locks):
                if b.can_lock(c,num,effective_locks):
                    out.append((c,num,True))
                else:
                    out.append((c,num,False))
        return out
    def step(self,action):
        if self.done:raise RuntimeError('Game finished')
        if action is not None and action not in self.legal():raise ValueError(f'Illegal action: {action}')
        who=self.actor
        if action is not None:
            c,num,lock=action;self.boards[who].rows[c].append(num)
            if lock:
                self.locks.add(c)
                if self.phase<self.n:self.white_locks.add(c)
            if who==self.active:self.active_marks+=1
        if self.phase==self.n and self.active_marks==0:
            self.boards[self.active].penalties+=1
        if self.phase<self.n:
            self.phase+=1
            if self.phase==self.n:self.white_locks.clear()
        if self.done:return
        if self.phase<self.n:pass
        else:
            self.turn+=1;self.active=(self.active+1)%self.n;self._roll()
