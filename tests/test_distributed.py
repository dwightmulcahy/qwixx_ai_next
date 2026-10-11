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
