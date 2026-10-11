import pytest
from qwixx_ai.distributed import Coordinator

def test_independent_sessions(tmp_path,monkeypatch):
    c=Coordinator()
    first=c.new_session();second=c.new_session()
    assert first['session_id']!=second['session_id']
    first_after=c.session_action(first['session_id'],None)
    assert first_after['session_id']==first['session_id']
    assert c.sessions[second['session_id']][0].view()=={k:v for k,v in second.items() if k!='session_id'}
    with pytest.raises(ValueError):c.session_action('nonexistent',None)

def test_policy_export_shape():
    c=Coordinator();p=c.policy_export()
    assert len(p['layers'])==3
    assert len(p['layers'][0]['w'][0])==p['observation_size']
    assert len(p['layers'][-1]['b'])==89


def test_browser_worker_uses_web_worker_messages():
    from qwixx_ai.distributed import WORKER_JS
    assert "onmessage=e=>" in WORKER_JS
    assert "postMessage({type:'progress'" in WORKER_JS


def test_training_database_history(tmp_path, monkeypatch):
    from qwixx_ai import distributed
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    with c.lock:
        c._record('central',[1,2,3,4],None)
    assert c.history(10)[0]['scores']==[1,2,3,4]
    assert c.stats()['database_games']==1
    c.db.close()


def test_worker_rate_limit_logic():
    from qwixx_ai.distributed import Handler
    class Fake:
        client_address=('192.0.2.42',1234)
        rate_lock=Handler.rate_lock
        rate_buckets={}
        limited=Handler.limited
    f=Fake()
    assert all(not f.limited() for _ in range(120))
    assert f.limited()


def test_priority_replay_alignment_and_exploration(tmp_path,monkeypatch):
    from qwixx_ai import distributed
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    sample=(None,0,0.0,None,False,None)
    c._add_transitions([sample,sample,sample])
    assert len(c.priorities)==len(c.replay)==3
    assert c.exploration()==0.30
    c.updates=100000
    assert 0.04<=c.exploration()<0.30
    c.db.close()


def test_replay_restores_from_saved_deterministic_games(tmp_path,monkeypatch):
    from qwixx_ai import distributed
    from qwixx_ai.core import Game
    from qwixx_ai.env import encode
    from qwixx_ai.agents import heuristic
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    g=Game(seed=7);moves=[]
    while not g.done:
        action=heuristic(g)
        moves.append(encode(action));g.step(action)
    with c.lock:c._record('browser',g.scores(),moves,7)
    c.db.close()
    restored=distributed.Coordinator()
    assert len(restored.replay)>0
    assert len(restored.priorities)==len(restored.replay)
    restored.db.close()
