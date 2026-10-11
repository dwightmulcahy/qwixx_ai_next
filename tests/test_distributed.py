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
