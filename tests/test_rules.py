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
    g.step((1,12,True))
    for _ in range(g.n-2):g.step(None)
    assert g.done

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


# Rulebook regression suite: white phase, colored phase, locks, ending and scoring.
def test_white_sum_available_to_all_players_and_optional():
    g=Game(seed=7);g.dice=(2,5,1,1,1,1)
    for seat in range(g.n):
        assert g.actor==seat
        assert (seat,7,False) in g.legal()
        g.step(None if seat%2 else (seat,7,False))
    assert g.phase==g.n and g.actor==g.active==0
    assert g.boards[1].penalties==0 and g.boards[3].penalties==0


def test_colored_phase_active_only_and_either_white_die():
    g=Game(seed=8);g.dice=(1,6,2,3,4,5)
    for _ in range(g.n):g.step(None)
    assert g.actor==0 and g.phase==g.n
    assert (0,3,False) in g.legal() and (0,8,False) in g.legal()
    assert (1,4,False) in g.legal() and (1,9,False) in g.legal()
    g.step((0,3,False))
    assert g.boards[0].rows[0]==[3]
    assert all(not b.rows[0] for b in g.boards[1:])


def test_clockwise_active_rotation():
    g=Game(seed=9)
    for expected in range(1,6):
        for _ in range(g.n+1):g.step(None)
        assert g.active==expected%g.n
        assert g.phase==0


def test_left_to_right_skip_is_permanent_both_directions():
    b=Board()
    assert b.can(0,8,set()) and b.can(2,7,set())
    b.rows[0].append(8);b.rows[2].append(7)
    for n in range(2,9):assert not b.can(0,n,set())
    for n in range(7,13):assert not b.can(2,n,set())
    assert b.can(0,9,set()) and b.can(2,6,set())


def test_active_penalty_only_when_both_phases_pass():
    g=Game(seed=10);g.dice=(1,2,2,2,2,2)
    g.step((0,3,False))
    for _ in range(g.n):g.step(None)
    assert g.boards[0].penalties==0
    g=Game(seed=10);g.dice=(1,2,2,2,2,2)
    for _ in range(g.n):g.step(None)
    g.step((0,3,False))
    assert g.boards[0].penalties==0
    g=Game(seed=10)
    for _ in range(g.n+1):g.step(None)
    assert g.boards[0].penalties==1
    assert all(b.penalties==0 for b in g.boards[1:])


def test_lock_closes_color_for_everyone_and_removes_colored_die():
    g=Game(seed=11);g.dice=(6,6,6,6,6,6)
    g.boards[0].rows[0]=[2,3,4,5,6]
    g.step((0,12,True))
    assert g.available_colors==(1,2,3)
    assert g.boards[0].score(g.locks)==28
    for _ in range(g.n-1):
        assert all(a[0]!=0 for a in g.legal())
        g.step(None)
    assert all(a[0]!=0 for a in g.legal())
    assert (1,12,False) not in g.legal()


def test_fourth_penalty_ends_immediately():
    g=Game(seed=12);g.boards[0].penalties=3
    for _ in range(g.n):g.step(None)
    assert not g.done
    g.step(None)
    assert g.done and g.boards[0].penalties==4
    with pytest.raises(RuntimeError):g.step(None)


def test_second_lock_ends_immediately():
    g=Game(seed=13);g.dice=(6,6,6,6,6,6)
    g.locks.add(0);g.boards[0].rows[1]=[2,3,4,5,6]
    g.step((1,12,True))
    for _ in range(g.n-1):g.step(None)
    assert g.done and g.locks=={0,1}
    with pytest.raises(RuntimeError):g.step(None)


def test_score_triangular_padlock_penalties_and_winner():
    a=Board();b=Board()
    a.rows[0]=[2,3,4,5,6,12];a.rows[2]=[12,10,9]
    a.penalties=2
    b.rows[1]=[2,3,4,5]
    assert a.score({0})==28+6-10
    assert b.score({0})==10
    assert a.score({0})>b.score({0})


def test_white_sum_remains_available_in_other_rows_after_lock():
    g=Game(seed=14);g.locks.add(0);g.dice=(3,4,1,1,1,1)
    assert (1,7,False) in g.legal()
    assert (0,7,False) not in g.legal()


def test_rulebook_faq_is_embedded_in_web_interface():
    from qwixx_ai.web import HTML
    for phrase in ('Qwixx rules & FAQ','White dice (everyone)','Colored die (active player only)',
                   'left to right','misthrow penalty','padlock','fourth misthrow penalty',
                   'highest score wins'):
        assert phrase in HTML


def test_color_dice_removed_and_lock_bonus():
    g=Game(seed=15)
    g.boards[0].rows[0]=[2,3,4,5,6]
    g.dice=(6,6,6,6,6,6)
    g.step((0,12,True))
    assert 0 not in g.available_colors
    assert g.boards[0].score(g.locks)==28


def test_double_dqn_model_shapes():
    from qwixx_ai.model import network
    import torch
    from qwixx_ai.env import observe
    model=network(len(observe(Game())))
    assert model(torch.from_numpy(observe(Game()))).shape==(89,)


def test_simultaneous_white_phase_allows_multiple_locks():
    g=Game(seed=42);g.dice=(6,6,1,1,1,1)
    for b in g.boards[:2]:b.rows[0]=[2,3,4,5,6]
    g.step((0,12,True))
    assert not g.done
    assert (0,12,True) in g.legal()
    g.step((0,12,True))
    assert g.boards[0].rows[0][-1]==12
    assert g.boards[1].rows[0][-1]==12
    for _ in range(g.n-2):g.step(None)
    assert 0 in g.locks
    assert 0 not in g.available_colors


def test_second_lock_waits_for_shared_white_phase():
    g=Game(seed=44);g.locks={0};g.dice=(6,6,1,1,1,1)
    g.boards[0].rows[1]=[2,3,4,5,6]
    g.step((1,12,True))
    assert not g.done
    for _ in range(g.n-1):g.step(None)
    assert g.done
