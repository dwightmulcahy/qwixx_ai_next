import pytest
from qwixx_ai.core import Game,Board
from qwixx_ai.env import encode,decode,observe,mask,QwixxEnv
from qwixx_ai.agents import heuristic
from qwixx_ai.tournament import play

def test_scoring_and_lock():
    b=Board();b.rows[0]=[2,3,4,5,6,12]
    assert b.score({0})==28
    assert b.score(set())==21

def test_lock_needs_five_prior():
    b=Board();assert not b.can_lock(0,12,set())
    b.rows[0]=[2,3,4,5,6];assert b.can_lock(0,12,set())

def test_reverse_row():
    b=Board();b.rows[2]=[12,10];assert b.can(2,9,set());assert not b.can(2,11,set())

def test_action_roundtrip():
    for i in range(89):assert encode(decode(i))==i

def test_dice_choices_both_whites():
    g=Game(seed=1);g.phase=g.n;g.dice=(1,6,2,3,4,5)
    assert (0,3,False) in g.legal()
    assert (0,8,False) in g.legal()

def test_active_can_mark_twice():
    g=Game(seed=1);g.dice=(1,2,2,2,2,2)
    g.step((0,3,False))
    for _ in range(g.n-1):g.step(None)
    assert g.actor==0 and g.phase==g.n
    g.step((1,3,False))
    assert g.boards[0].rows[0]==[3] and g.boards[0].rows[1]==[3]
    assert g.boards[0].penalties==0

def test_penalty_only_if_no_active_marks():
    g=Game(seed=1)
    for _ in range(g.n+1):g.step(None)
    assert g.boards[0].penalties==1

def test_lock_ends_after_two():
    g=Game(seed=1)
    g.boards[0].rows[0]=[2,3,4,5,6]
    g.dice=(6,6,6,6,6,6)
    g.step((0,12,True));assert 0 in g.locks
    g.boards[1].rows[1]=[2,3,4,5,6]
    assert (1,12,True) in g.legal()
    g.step((1,12,True));assert g.done

def test_env_finishes():
    e=QwixxEnv();o,i=e.reset(seed=17)
    assert len(o)==len(observe(e.game))
    for _ in range(1000):
        legal=i['action_mask'].nonzero()[0]
        o,r,done,trunc,i=e.step(int(legal[0]))
        if done:break
    assert done

def test_deterministic_game():
    a=Game(seed=3);b=Game(seed=3)
    for _ in range(1000):
        assert a.dice==b.dice and a.actor==b.actor
        if a.done:break
        action=heuristic(a);a.step(action);b.step(action)
    assert a.scores()==b.scores()

def test_tournament():
    scores,shares=play([heuristic]*4,seed=5)
    assert len(scores)==4 and abs(sum(shares)-1)<1e-9


def test_final_number_requires_five_prior_marks():
    for color in range(4):
        b=Board()
        last=12 if color<2 else 2
        assert not b.can(color,last,set())
        b.rows[color]=list(range(2,7)) if color<2 else list(range(12,7,-1))
        assert b.can(color,last,set())
        assert b.can_lock(color,last,set())


def test_final_number_only_available_with_lock_action():
    g=Game(seed=4)
    g.dice=(6,6,6,6,6,6)
    assert (0,12,False) not in g.legal()
    assert (0,12,True) not in g.legal()
    g.boards[0].rows[0]=[2,3,4,5,6]
    assert (0,12,True) in g.legal()
    assert (0,12,False) not in g.legal()
    with pytest.raises(ValueError):
        g.step((0,12,False))
    g.step((0,12,True))
    assert 0 in g.locks
