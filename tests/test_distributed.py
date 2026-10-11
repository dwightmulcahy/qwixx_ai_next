import os
import subprocess
import sys

import pytest
from qwixx_ai.distributed import Coordinator

def test_default_training_data_path_is_writable(tmp_path):
    env=os.environ.copy();env.pop('QWIXX_DATA',None)
    result=subprocess.run(
        [sys.executable,'-c','from qwixx_ai import distributed; print(distributed.ROOT)'],
        cwd=tmp_path,env=env,text=True,capture_output=True,check=True,
    )
    assert result.stdout.strip()==str(tmp_path/'runs')
    assert (tmp_path/'runs').is_dir()


def test_explicit_training_data_path_is_honored(tmp_path):
    target=tmp_path/'custom-runs';env=os.environ.copy();env['QWIXX_DATA']=str(target)
    result=subprocess.run(
        [sys.executable,'-c','from qwixx_ai import distributed; print(distributed.ROOT)'],
        env=env,text=True,capture_output=True,check=True,
    )
    assert result.stdout.strip()==str(target)
    assert target.is_dir()


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


def test_checkpoint_restart_restores_optimizer_and_counters(tmp_path,monkeypatch):
    from qwixx_ai import distributed
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    c.updates=17;c.games=23;c.version=5
    c.checkpoint()
    c.db.close()
    restored=distributed.Coordinator()
    assert restored.updates==17
    assert restored.games==23
    assert restored.version==5
    assert (tmp_path/'latest.pt').exists()
    assert (tmp_path/'trainer.pt').exists()
    restored.db.close()


def test_difficulty_selection_and_explanations():
    from qwixx_ai.web import Session,HTML
    assert 'id="difficulty"' in HTML
    for level in ('beginner','intermediate','advanced','champion'):
        session=Session(difficulty=level)
        view=session.view()
        assert view['difficulty']==level
        assert all(x['skipped']>=0 and x['immediate']>0 for x in view['explanations'])
    with pytest.raises(ValueError):Session(difficulty='impossible')


def test_training_pause_and_resume(tmp_path,monkeypatch):
    from qwixx_ai import distributed
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    assert not c.stats()['paused']
    c.paused.set()
    assert c.stats()['paused']
    c.paused.clear()
    assert not c.stats()['paused']
    c.db.close()


def test_browser_python_deterministic_pass_parity(tmp_path,monkeypatch):
    import json,shutil
    from qwixx_ai import distributed
    if shutil.which('node') is None:pytest.skip('Node.js unavailable')
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    job=c.ticket();job['rolls']=distributed.rolls_for(job['seed'])
    js=r'''
const vm=require('vm');let input='';process.stdin.on('data',c=>input+=c);
process.stdin.on('end',async()=>{
 const {source,job}=JSON.parse(input);let submitted=null;
 const context={console,Math,setTimeout,postMessage:()=>{},fetch:async(url,opts)=>{
  if(url==='/api/policy')return {ok:true,json:async()=>({version:0,layers:[{w:Array.from({length:89},()=>[]),b:Array(89).fill(0)}]})};
  if(url==='/api/job')return {ok:true,json:async()=>job};
  if(url==='/api/submit'){submitted=JSON.parse(opts.body);return {ok:true,json:async()=>({accepted:true})}};
  throw Error('unexpected '+url);
 }};
 vm.createContext(context);vm.runInContext(source,context);
 await vm.runInContext('runGame()',context);
 process.stdout.write(JSON.stringify(submitted));
});
'''
    output=subprocess.run(['node','-e',js],input=json.dumps({'source':distributed.WORKER_JS,'job':job}),text=True,capture_output=True,check=True,timeout=30)
    moves=json.loads(output.stdout)
    assert moves['ticket']==job['ticket']
    assert c.accept(moves)['accepted']
    c.db.close()


def test_central_rollout_generates_valid_persisted_game(tmp_path,monkeypatch):
    from qwixx_ai import distributed
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    c.central_rollout()
    assert c.central_games==1
    assert c.stats()['database_games']==1
    assert len(c.replay)==len(c.priorities)>0
    c.db.close()


def test_admin_endpoint_requires_token(tmp_path,monkeypatch):
    import json,threading,urllib.request,urllib.error
    from http.server import ThreadingHTTPServer
    from qwixx_ai import distributed
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    monkeypatch.setattr(distributed,'ADMIN_TOKEN','test-secret')
    coord=distributed.Coordinator()
    class TestHandler(distributed.Handler):
        def log_message(self,*args):pass
    TestHandler.coord=coord
    server=ThreadingHTTPServer(('127.0.0.1',0),TestHandler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/api/admin'
    def request(token=None,command='pause'):
        headers={'Content-Type':'application/json'}
        if token:headers['X-Admin-Token']=token
        return urllib.request.urlopen(urllib.request.Request(url,data=json.dumps({'command':command}).encode(),headers=headers),timeout=5)
    try:
        with pytest.raises(urllib.error.HTTPError) as err:request()
        assert err.value.code==403
        with request('test-secret') as response:assert json.load(response)['paused']
        with request('test-secret','resume') as response:assert not json.load(response)['paused']
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5);coord.db.close()
