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


def test_optimal_opening_roll_supports_two_edge_marks():
    from qwixx_ai.core import Game
    from qwixx_ai.web import opening_supports_two_edge_marks, optimal_opening_game

    good=Game(seed=1);good.dice=(1,1,2,2,2,2)
    good.active=2
    bad=Game(seed=1);bad.dice=(3,4,3,3,3,3)
    assert opening_supports_two_edge_marks(good)
    assert not opening_supports_two_edge_marks(bad)
    found=optimal_opening_game()
    assert opening_supports_two_edge_marks(found)


def test_play_session_randomizes_first_roller_without_changing_core_game_default():
    from qwixx_ai.core import Game
    from qwixx_ai.web import randomize_starting_player

    assert Game(seed=1).active==0
    starters=set()
    for seed in range(64):
        game=randomize_starting_player(Game(seed=seed))
        assert 0<=game.active<game.n
        starters.add(game.active)
    assert starters=={0,1,2,3}


def test_session_random_seating_shuffles_ai_identities_and_can_be_disabled():
    from qwixx_ai.web import AI_NAMES, Session

    session=Session()
    session.reset_game(randomize_seating=False)
    assert session.game.active==0
    assert session.ai_names==list(AI_NAMES)

    seen=set()
    for _ in range(40):
        session.reset_game(randomize_seating=True)
        assert sorted(session.ai_names)==sorted(AI_NAMES)
        seen.add(tuple(session.ai_names))
    assert len(seen)>1


def test_independent_sessions(tmp_path,monkeypatch):
    from qwixx_ai.web import opening_supports_two_edge_marks
    c=Coordinator()
    first=c.new_session();second=c.new_session()
    assert first['session_id']!=second['session_id']
    optimal=c.new_session(optimal_first_roll=True,randomize_seating=False)
    assert opening_supports_two_edge_marks(c.sessions[optimal['session_id']][0].game)
    first_after=c.session_action(first['session_id'],None)
    assert first_after['session_id']==first['session_id']
    assert c.sessions[second['session_id']][0].view()=={k:v for k,v in second.items() if k!='session_id'}
    with pytest.raises(ValueError):c.session_action('nonexistent',None)

def test_policy_export_shape():
    c=Coordinator();p=c.policy_export()
    assert p['architecture']=='dueling-v1'
    assert len(p['trunk'])==3
    assert len(p['trunk'][0]['w'][0])==p['observation_size']
    assert len(p['value']['b'])==1
    assert len(p['advantage']['b'])==89


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


def test_back_to_step1_is_not_offered_after_no_move_auto_pass():
    from qwixx_ai.web import Session

    session=Session()
    session.ai=lambda game: None
    game=session.game
    game.active=0;game.phase=0;game.turn=4;game.dice=(6,6,1,1,1,1)
    game.boards[0].rows[2]=[11]
    game.boards[0].rows[3]=[11]
    assert game.legal()==[]

    session.apply_action(None,'human')
    session.advance()

    assert session.game.phase==session.game.n
    assert session.game.actor==0
    assert session.step1_snapshot is not None
    assert session.step1_snapshot[0].legal()==[]
    assert not session.can_undo_step1()
    assert session.view()['can_undo_step1'] is False


def test_back_to_step1_remains_available_after_voluntary_pass_with_legal_moves():
    from qwixx_ai.web import Session

    session=Session()
    session.ai=lambda game: None
    game=session.game
    game.active=0;game.phase=0;game.turn=4;game.dice=(1,1,1,1,1,1)
    assert game.legal()

    session.apply_action(None,'human')
    session.advance()

    assert session.game.phase==session.game.n
    assert session.can_undo_step1()
    session.undo_step1()
    assert session.game.phase==0
    assert session.game.actor==0
    assert session.game.legal()


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


def test_training_roster_has_one_rotating_learner_and_larger_league(tmp_path,monkeypatch):
    from qwixx_ai import distributed
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    c.rng.seed(1234)
    roster=c._training_roster()
    assert len(roster)==4
    assert sum(kind=='current' for kind,_ in roster)==1
    assert c.league.maxlen==12
    c.db.close()


def test_central_rollout_keeps_each_seat_policy_stable_for_whole_game(tmp_path,monkeypatch):
    from qwixx_ai import distributed
    from qwixx_ai.heuristic_v2 import heuristic_v2
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    roster=[(f'seat-{i}',None) for i in range(4)]
    seen={i:set() for i in range(4)}
    monkeypatch.setattr(c,'_training_roster',lambda:roster)

    def action(spec,game):
        seen[game.actor].add(spec[0])
        return heuristic_v2(game)

    monkeypatch.setattr(c,'_training_action',action)
    c.central_rollout()
    assert all(seen[i]=={f'seat-{i}'} for i in range(4))
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


def test_legacy_checkpoint_starts_fresh_without_deleting_files(tmp_path, monkeypatch):
    import torch
    from qwixx_ai import distributed
    from qwixx_ai.model import legacy_network
    monkeypatch.setattr(distributed, 'ROOT', tmp_path)
    old = legacy_network(195)
    torch.save({'model': old.state_dict(), 'obs_size': 195}, tmp_path / 'latest.pt')
    torch.save({'model': old.state_dict(), 'obs_size': 195, 'win_share': 0.8}, tmp_path / 'best.pt')
    coordinator = distributed.Coordinator()
    assert coordinator.model[0].in_features == coordinator.obs_size
    assert coordinator.target[0].in_features == coordinator.obs_size
    assert coordinator.best_model is None
    assert (tmp_path / 'latest.pt').exists()
    assert (tmp_path / 'best.pt').exists()
    coordinator.db.close()


def test_same_size_legacy_policy_is_rejected_for_live_training(tmp_path,monkeypatch):
    import torch
    from qwixx_ai import distributed
    from qwixx_ai.env import observe
    from qwixx_ai.core import Game
    from qwixx_ai.model import legacy_network
    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    size=len(observe(Game()))
    old=legacy_network(size)
    torch.save({'model':old.state_dict(),'obs_size':size},tmp_path/'latest.pt')
    coordinator=distributed.Coordinator()
    assert coordinator.model.architecture=='dueling-v1'
    assert coordinator.model is not old
    coordinator.db.close()


def test_browser_worker_observation_matches_python_dimensions():
    from qwixx_ai import distributed
    from qwixx_ai.core import Game
    from qwixx_ai.env import observe
    assert len(observe(Game())) == 262
    assert 'remaining/11' in distributed.WORKER_JS
    assert "seats=Array.from({length:4},(_,i)=>(who+i)%4)" in distributed.WORKER_JS
    assert "((active-who+4)%4)/4" in distributed.WORKER_JS


def test_actor_relative_observation_starts_with_current_actor_board():
    import numpy as np
    from qwixx_ai.core import Game
    from qwixx_ai.env import actor_seat_order, observe

    game=Game(seed=13)
    game.active=1
    game.phase=1
    assert game.actor==2
    game.boards[0].rows[0]=[2]
    game.boards[1].rows[0]=[3]
    game.boards[2].rows[0]=[4]
    game.boards[3].rows[0]=[5]
    assert actor_seat_order(game)==(2,3,0,1)
    obs=observe(game)
    first_row=obs[:11]
    expected=np.array([float(n==4) for n in range(2,13)],dtype=np.float32)
    assert np.array_equal(first_row,expected)
    # actor identity is no longer encoded as an absolute seat feature.
    assert len(obs)==262


def test_training_dashboard_link_and_status():
    from qwixx_ai.distributed import DASHBOARD_HTML
    from qwixx_ai.web import HTML

    assert 'href="/training"' in HTML
    assert 'id="training-dashboard"' in DASHBOARD_HTML
    assert "Training ${x.paused?'PAUSED':'ACTIVE'}" in DASHBOARD_HTML


def test_dialog_player_names_use_registered_identity_with_guest_fallback():
    from qwixx_ai.web import HTML

    for expected in (
        "function humanDialogName()",
        "function dialogPlayerName(i)",
        "verifiedPlayer&&playMode()==='registered'&&verifiedPlayer.name",
        ":'Guest'",
        "rankings=state.scores.map((score,i)=>({name:dialogPlayerName(i),score}))",
        "tie for 1st!",
    ):
        assert expected in HTML, expected
    assert "rankings=state.scores.map((score,i)=>({name:playerNames[i],score}))" not in HTML


def test_step_two_strike_warning_confirms_only_when_legal_marks_exist():
    from qwixx_ai.web import HTML

    assert "Passing now gives you a strike (−5 points)." in HTML
    assert "function shouldConfirmPenaltyPass()" in HTML
    assert "state.legal.length>0" in HTML
    assert "Take a missed-throw penalty?" in HTML
    assert "Pass anyway (−5)" in HTML
    assert "if(shouldConfirmPenaltyPass()){confirmPenaltyPass();return}act(null)" in HTML


def test_player_stats_route_accepts_plain_url_and_filters():
    """Stats requests from the UI must work without a query string."""
    import inspect
    from qwixx_ai.distributed import Handler

    source = inspect.getsource(Handler.do_GET)
    assert "self.path=='/api/player-stats' or self.path.startswith('/api/player-stats?')" in source


def test_ai_stats_route_is_public_and_supports_difficulty_filter():
    import inspect
    from qwixx_ai.distributed import Handler

    source = inspect.getsource(Handler.do_GET)
    assert "self.path=='/api/ai-stats' or self.path.startswith('/api/ai-stats?')" in source
    assert "self.coord.player_stats.ai_stats(query.get('difficulty',[None])[0])" in source


def test_game_tabs_are_present_and_preserve_game_nodes():
    from qwixx_ai.web import HTML

    for name in ("Play", "Account & Stats", "Top 10", "AI Stats", "SHOP", "Training", "Rules & FAQ", "About"):
        assert name in HTML
    assert "Move Analysis" not in HTML.split("const sections=", 1)[1].split(";", 1)[0]
    for node in ("status", "boards", "auth-login", "career", "training-dashboard", "gameplay-modal", "rules-faq", "shop-qwixx", "about-qwixx", "section-ai-stats"):
        assert node in HTML
    assert "window.showTab=activate" in HTML
    assert "activate('play')" in HTML
    assert "panels.play.appendChild" in HTML
    assert "panels.rules.appendChild(rules)" in HTML
    assert "panels.shop.appendChild(shop)" in HTML
    assert "panels.about.appendChild(about)" in HTML
    assert "if(id==='ai-stats'&&typeof showAiStats==='function')showAiStats();" in HTML


def test_reset_endpoint_supports_optimal_first_roll():
    import inspect
    from qwixx_ai.distributed import Coordinator, Handler

    handler=inspect.getsource(Handler.do_POST)
    coordinator=inspect.getsource(Coordinator.new_session)
    assert "optimal=data.get('optimal_first_roll',False)" in handler
    assert "randomize=data.get('randomize_seating',True)" in handler
    assert "if type(optimal) is not bool:raise ValueError('optimal_first_roll must be boolean')" in handler
    assert "if type(randomize) is not bool:raise ValueError('randomize_seating must be boolean')" in handler
    assert "optimal_first_roll=False,randomize_seating=True" in coordinator
    assert "session.reset_game(optimal_first_roll,randomize_seating)" in coordinator


def test_guest_mode_overrides_registered_cookie_without_logging_out():
    import inspect
    from qwixx_ai.distributed import Handler

    source = inspect.getsource(Handler)
    assert "guest=data.get('guest',False)" in source
    assert "None if guest else self.player_id()" in source
    assert "if type(guest) is not bool:raise ValueError('guest must be boolean')" in source
    # Guest sessions intentionally have no player_id, so an existing auth cookie
    # must not make their moves/progress fail ownership checks.
    assert "entry[2] is not None and entry[2]!=self.player_id()" in source
    assert "entry[2] is not None and entry[2] != self.player_id()" in source


def test_distributed_training_dashboard_remains_embedded():
    from qwixx_ai.distributed import DASHBOARD_HTML

    assert 'id="training-dashboard"' in DASHBOARD_HTML
    assert 'id="trainingStats"' in DASHBOARD_HTML
    assert 'id="trainingHistory"' in DASHBOARD_HTML
    assert "window.showTab=activate" in DASHBOARD_HTML


def test_game_script_turn_controls_have_valid_concat():
    from qwixx_ai.web import HTML

    assert 'class="move-status"' in HTML
    assert 'class="dice-move-controls"' in HTML
    assert 'class="move-buttons"' in HTML
    assert 'onclick="confirmPhase()"' in HTML
    assert 'function maybeAutoPass()' in HTML
    assert 'id="decision-modal"' in HTML
    assert "function confirmLockAction(action)" in HTML
    assert "function confirmPenaltyPass()" in HTML
    assert "AUTO_PASS_KEY='qwixx-auto-pass'" in HTML
    assert 'id="pref-auto-pass"' in HTML
    assert '⚡ AUTO PASS · No legal marks available' in HTML
    assert 'No legal marks · Pass required' in HTML
    assert 'function render()' in HTML
    assert 'async function reset(optimalFirstRoll=false)' in HTML
    assert 'function openNewGameDialog()' in HTML
    assert 'Find optimal first roll' in HTML


def test_colorful_favicon_linked_and_available():
    from qwixx_ai.favicon import FAVICON_SVG
    from qwixx_ai.web import HTML

    assert '<link rel="icon" href="/favicon.svg" type="image/svg+xml">' in HTML
    assert FAVICON_SVG.startswith('<svg')
    for color in ('#ed3345', '#f8bd20', '#1ba45c', '#3178ed'):
        assert color in FAVICON_SVG


def test_account_tab_unhides_career_and_displays_full_game_history():
    from qwixx_ai.web import HTML

    assert "panel.hidden=false;" in HTML
    assert "if(id==='account'&&typeof showStats==='function')showStats()" in HTML
    for field in ("Current win streak", "Longest win streak", "Highest score", "Lowest score",
                  "All player scores", "Results by AI difficulty", "Recent completed games",
                  "Average penalties", "Average turns", "Average place", "Win rate"):
        assert field in HTML
    assert "/api/player-stats?limit=100" in HTML


def test_unavailable_white_total_skips_without_confirmation():
    from qwixx_ai.web import HTML

    assert "Pass on the white dice total" not in HTML
    assert "No legal marks available. Skipping gives you a strike" not in HTML
    assert "function skipPhase()" in HTML
    assert "if(shouldConfirmPenaltyPass()){confirmPenaltyPass();return}act(null)" in HTML
    assert "A strike only occurs if you pass both opportunities on your roll." in HTML


def test_auto_pass_only_targets_human_phases_with_zero_legal_moves():
    from qwixx_ai.web import HTML

    assert "state.actor===0&&state.legal.length===0" in HTML
    assert "preference(AUTO_PASS_KEY,'on')!=='on'" in HTML
    assert "state.actor!==0||state.legal.length!==0" in HTML
    assert "autoPassTimer=setTimeout" in HTML
    assert "act(null)" in HTML


def test_lock_confirmation_warns_when_second_lock_would_end_game():
    from qwixx_ai.web import HTML

    for expected in (
        "function projectedLockScores(action)",
        "function remainingWhiteCanChange(action)",
        "secondLock=newDistinct&&state.locks.length===1",
        "This is the second locked row and ends the game.",
        "This creates the second locked row, so the game will end after the remaining white-dice decisions.",
        "Projected final scores",
        "Possible final scores",
    ):
        assert expected in HTML, expected
    assert "window.confirm('Close this row?" not in HTML


def test_penalty_notifications_block_only_for_the_human_player():
    from qwixx_ai.web import HTML

    assert "function clearPenaltyModal()" in HTML
    assert "function showNewPenalties(previous,next)" in HTML
    assert "previous.boards||!next.boards||next.done" in HTML
    assert "penaltySession=next.session_id||previous.session_id" in HTML
    assert "penaltySession!==state.session_id" in HTML
    assert "board.penalties>before" in HTML
    assert "{player:i,name:dialogPlayerName(i),count:board.penalties-before}" in HTML
    assert "humanChanges=changes.filter(change=>change.player===0)" in HTML
    assert "if(!humanChanges.length)return" in HTML
    assert "for(const change of humanChanges)" in HTML
    assert "change.name+' received '+change.count+' X'" in HTML
    assert "' points).'" in HTML
    assert "takes '+(strikes===1?'a strike':strikes+' strikes')+' · −'" in HTML
    assert "showNewPenalties(previous,next)" in HTML
    assert "overlay.setAttribute('role','alertdialog')" in HTML
    assert "showGameOver(){if(gameOverShown||!state.done)return;clearPenaltyModal()" in HTML
    assert "async function startNewGame(optimalFirstRoll){closeNewGameDialog();clearPenaltyModal()" in HTML
    assert "async function reset(optimalFirstRoll=false){clearPenaltyModal()" in HTML
    assert "if(next&&state?.session_id===sid)" in HTML


def test_live_game_does_not_hold_coordinator_lock_while_ai_advances(tmp_path,monkeypatch):
    import threading
    from types import SimpleNamespace
    from qwixx_ai import distributed

    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    acquired=[]

    class FakeSession:
        def __init__(self):
            self.lock=threading.Lock()
            self.game=SimpleNamespace(done=False)
            self.difficulty='champion'
            self.ai_names=[]
        def apply_action(self,action,source):
            pass
        def advance(self):
            result=[]
            def try_lock():
                ok=c.lock.acquire(timeout=.25)
                result.append(ok)
                if ok:c.lock.release()
            th=threading.Thread(target=try_lock)
            th.start();th.join()
            acquired.extend(result)
        def view(self):
            return {'done':False}

    session=FakeSession()
    c.sessions['live']=(session,0.0,None)
    try:
        c.session_action('live',None)
        assert acquired==[True]
    finally:
        c.db.close()


def test_checkpoint_defers_heavy_evaluation_during_recent_game(tmp_path,monkeypatch):
    import time
    from qwixx_ai import distributed

    monkeypatch.setattr(distributed,'ROOT',tmp_path)
    c=distributed.Coordinator()
    c.sessions['live']=(object(),time.time(),None)
    try:
        assert c.checkpoint() is False
        assert not (tmp_path/'latest.pt').exists()
    finally:
        c.db.close()


def test_compose_honors_interactive_training_env():
    from pathlib import Path

    compose=Path('compose.yaml').read_text()
    assert 'QWIXX_CENTRAL_WORKERS: ${QWIXX_CENTRAL_WORKERS:-0}' in compose
    assert 'QWIXX_TORCH_THREADS: ${QWIXX_TORCH_THREADS:-1}' in compose
    assert 'QWIXX_TRAINING_UPDATES_PER_CYCLE: ${QWIXX_TRAINING_UPDATES_PER_CYCLE:-1}' in compose
    assert 'QWIXX_TRAINING_CYCLE_SLEEP: ${QWIXX_TRAINING_CYCLE_SLEEP:-0.20}' in compose
    assert 'QWIXX_CENTRAL_WORKERS: "2"' not in compose


def test_browser_worker_limit_is_configurable():
    from qwixx_ai import distributed

    assert 'Math.min(2,navigator.hardwareConcurrency||1)' in distributed.DASHBOARD_HTML
    assert 'Up to 2 worker threads' in distributed.DASHBOARD_HTML


def test_live_champion_uses_bounded_search_policy(monkeypatch):
    from qwixx_ai import web

    session=web.Session(model=object(),difficulty='champion')
    expected=('green', 2, False)
    calls=[]

    def fake_live_champion(game):
        calls.append(game)
        return expected

    monkeypatch.setattr(web,'live_champion',fake_live_champion)

    assert session.ai(session.game)==expected
    assert calls==[session.game]


def test_live_champion_does_not_depend_on_promoted_model(monkeypatch):
    from qwixx_ai import web

    session=web.Session(model=None,difficulty='champion')
    expected=('red', 2, False)

    monkeypatch.setattr(web,'live_champion',lambda game:expected)

    assert session.ai(session.game)==expected


def test_ai_thinking_snapshot_exposes_current_roll_before_decision():
    from qwixx_ai.core import Game
    from qwixx_ai.web import Session

    session=Session()
    session.game=Game(seed=21)
    original_dice=session.game.dice
    session.game.active=1
    session.game.phase=0
    seen=[]

    def fake_ai(game):
        assert session.thinking_actor==game.actor
        assert session.thinking_view is not None
        seen.append((
            tuple(session.thinking_view["dice"]),
            session.thinking_view["roller"],
            session.thinking_view["actor"],
        ))
        return None

    session.ai=fake_ai
    session.advance()

    assert seen
    assert seen[0][0]==tuple(original_dice)
    assert seen[0][1]==1
    assert session.thinking_view is None
    assert session.thinking_actor is None


def test_runtime_http_routes_cover_success_and_error_paths(tmp_path, monkeypatch):
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from qwixx_ai import distributed

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    monkeypatch.setattr(distributed, "BROWSER_TRAINING_ENABLED", False)
    monkeypatch.setattr(distributed, "ADMIN_TOKEN", "admin-secret")
    coord = distributed.Coordinator()

    class TestHandler(distributed.Handler):
        def log_message(self, *args):
            pass

    TestHandler.coord = coord
    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path, headers=None):
        return urllib.request.urlopen(
            urllib.request.Request(base + path, headers=headers or {}), timeout=5
        )

    def post(path, payload=None, headers=None):
        body = json.dumps(payload if payload is not None else {}).encode()
        merged = {"Content-Type": "application/json", **(headers or {})}
        return urllib.request.urlopen(
            urllib.request.Request(base + path, data=body, headers=merged), timeout=5
        )

    try:
        with get("/health") as response:
            assert json.load(response) == {"healthy": True}
        with get("/api/stats") as response:
            assert "model_version" in json.load(response)
        with get("/api/history?limit=bogus") as response:
            assert json.load(response)["games"] == []
        with get("/api/auth/me") as response:
            assert json.load(response) == {"player": None}
        with get("/api/policy") as response:
            assert "observation_size" in json.load(response)
        with get("/favicon.svg") as response:
            assert response.headers["Content-Type"] == "image/svg+xml"
        with get("/") as response:
            assert "Browser training disabled" in response.read().decode()
        with get("/training") as response:
            assert "Browser training disabled" in response.read().decode()

        for path in ("/worker.js", "/api/job"):
            with pytest.raises(urllib.error.HTTPError) as err:
                get(path)
            assert err.value.code == 403

        with pytest.raises(urllib.error.HTTPError) as err:
            get("/api/player-stats")
        assert err.value.code == 401
        with pytest.raises(urllib.error.HTTPError) as err:
            get("/api/session-progress?session_id=missing")
        assert err.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as err:
            get("/does-not-exist")
        assert err.value.code == 404

        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/admin", {"command": "pause"})
        assert err.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/admin", {"command": "unknown"},
                 {"X-Admin-Token": "admin-secret"})
        assert err.value.code == 400

        for payload in (
            {"guest": "yes"},
            {"guest": True, "optimal_first_roll": "yes"},
            {"guest": True, "randomize_seating": "yes"},
        ):
            with pytest.raises(urllib.error.HTTPError) as err:
                post("/reset", payload)
            assert err.value.code == 400

        with post("/reset", {
            "guest": True,
            "difficulty": "intermediate",
            "optimal_first_roll": False,
            "randomize_seating": False,
        }) as response:
            session = json.load(response)
        assert session["session_id"]
        assert session["difficulty"] == "intermediate"

        with pytest.raises(urllib.error.HTTPError) as err:
            post("/move", {
                "session_id": "missing",
                "action": None,
                "expected_turn": 0,
                "expected_phase": 0,
            })
        assert err.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as err:
            post("/not-a-route", {})
        assert err.value.code == 404

        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/submit", {"ticket": "x", "moves": []})
        assert err.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        coord.db.close()


def test_runtime_coordinator_session_and_ticket_error_paths(tmp_path, monkeypatch):
    import time

    from qwixx_ai import distributed

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    coord = distributed.Coordinator()
    try:
        ticket = coord.ticket()
        assert ticket["max_moves"] == 1000
        coord.pending[ticket["ticket"]] = (ticket["seed"], time.time() - 901)
        with pytest.raises(ValueError, match="Expired or unknown ticket"):
            coord.accept({"ticket": ticket["ticket"], "moves": []})

        with pytest.raises(ValueError, match="Invalid submission"):
            coord.accept({"ticket": 123, "moves": []})
        with pytest.raises(ValueError, match="Invalid submission"):
            coord.accept({"ticket": "x", "moves": "bad"})

        session = coord.new_session("intermediate", randomize_seating=False)
        sid = session["session_id"]
        current = coord.sessions[sid][0].game
        with pytest.raises(ValueError, match="Game state changed"):
            coord.session_action(sid, None, expected_turn=current.turn + 1)
        with pytest.raises(ValueError, match="Step already advanced"):
            coord.session_action(sid, None, expected_phase=current.phase + 1)
        with pytest.raises(ValueError, match="Session expired"):
            coord.session_action("missing", None)
        with pytest.raises(ValueError, match="Session expired"):
            coord.session_undo_step1("missing", 0, 0)
        with pytest.raises(ValueError, match="Game state changed"):
            coord.session_undo_step1(sid, None, current.phase)

        exported = coord.policy_export()
        assert exported["version"] == coord.version
        assert exported["observation_size"] == coord.obs_size
    finally:
        coord.db.close()


def test_runtime_search_validation_and_override_paths(monkeypatch):
    from qwixx_ai import runtime_search
    from qwixx_ai.core import Game

    game = Game(seed=17)
    with pytest.raises(ValueError, match="futures must be positive"):
        runtime_search.evaluate_actions(game, futures=0)

    finished = Game(seed=18)
    finished.locks = {0, 1}
    finished.phase = finished.n
    with pytest.raises(ValueError, match="finished game"):
        runtime_search.evaluate_actions(finished)

    with pytest.raises(ValueError, match="invalid search parameters"):
        runtime_search.decide(game, futures=0)
    with pytest.raises(ValueError, match="invalid turn window"):
        runtime_search.decide(game, start_turn=20, end_turn=10)

    teacher = ("red", 2, False)
    monkeypatch.setattr(runtime_search, "heuristic_v2", lambda g: teacher)
    monkeypatch.setattr(runtime_search, "eligible", lambda *args, **kwargs: False)
    assert runtime_search.decide(game)[0] == teacher

    monkeypatch.setattr(runtime_search, "eligible", lambda *args, **kwargs: True)
    challenger = ("yellow", 2, False)
    monkeypatch.setattr(
        runtime_search,
        "evaluate_actions",
        lambda *args, **kwargs: [
            {"action": teacher, "win_share": 0.4, "mean_score": 50},
            {"action": challenger, "win_share": 0.6, "mean_score": 55},
        ],
    )
    assert runtime_search.decide(game)[0] == challenger

    stats = {"decisions": 0, "v2_not_eligible": 0, "v2_safeguard": 0, "override": 0}
    policy = runtime_search.policy(stats=stats)
    assert policy(game) == challenger
    assert stats["decisions"] == 1
    assert stats["override"] == 1


def test_runtime_session_error_and_difficulty_branches(monkeypatch):
    from qwixx_ai import web
    from qwixx_ai.core import Game

    with pytest.raises(ValueError, match="Invalid difficulty"):
        web.Session(difficulty="impossible")

    session = web.Session(difficulty="beginner")
    beginner = ("red", 2, False)
    monkeypatch.setattr(web, "random_agent", lambda game: beginner)
    assert session.ai(session.game) == beginner

    session.difficulty = "intermediate"
    intermediate = ("yellow", 2, False)
    monkeypatch.setattr(web, "heuristic", lambda game: intermediate)
    assert session.ai(session.game) == intermediate

    session.difficulty = "champion"
    champion = ("green", 12, False)
    monkeypatch.setattr(web, "live_champion", lambda game: champion)
    assert session.ai(session.game) == champion

    session.difficulty = "advanced"
    session.model = None
    fallback = ("blue", 12, False)
    monkeypatch.setattr(web, "heuristic_v2", lambda game: fallback)
    assert session.ai(session.game) == fallback

    model = object()
    session.model = model
    chosen = ("red", 3, False)
    monkeypatch.setattr(
        web, "choose",
        lambda selected_model, game, epsilon: chosen
        if selected_model is model and epsilon == 0.18 else None,
    )
    assert session.ai(session.game) == chosen

    with pytest.raises(ValueError, match="optimal_first_roll must be boolean"):
        session.reset_game("yes")
    with pytest.raises(ValueError, match="randomize_seating must be boolean"):
        session.reset_game(False, "yes")

    done = Game(seed=22)
    done.locks = {0, 1}
    done.phase = done.n
    session.game = done
    with pytest.raises(ValueError, match="already finished"):
        session.apply_action(None, "human")

    ai_turn = Game(seed=23)
    ai_turn.active = 1
    ai_turn.phase = 0
    session.game = ai_turn
    with pytest.raises(ValueError, match="Not the human player turn"):
        session.apply_action(None, "human")

    human_turn = Game(seed=24)
    human_turn.active = 0
    human_turn.phase = 0
    session.game = human_turn
    with pytest.raises(ValueError, match="AI cannot act for human"):
        session.apply_action(None, "ai")

    session.step1_snapshot = None
    with pytest.raises(ValueError, match="Step 1 cannot be undone"):
        session.undo_step1()


def test_runtime_opening_helpers_and_history_trimming(monkeypatch):
    from qwixx_ai import web
    from qwixx_ai.core import Game

    game = Game(seed=31)
    game.rng.seed(7)
    assert 0 <= web.randomize_starting_player(game).active < game.n

    monkeypatch.setattr(web, "opening_supports_two_edge_marks", lambda game: False)
    with pytest.raises(RuntimeError, match="Unable to find an optimal opening roll"):
        web.optimal_opening_game(max_attempts=2, randomize_seating=False)

    normal = web.new_play_game(False, False)
    assert normal.active == 0

    session = web.Session(difficulty="intermediate")
    session.game = Game(seed=32)
    session.history = [{"n": i} for i in range(200)]
    action = session.game.legal()[0] if session.game.legal() else None
    session.apply_action(action, "human")
    assert len(session.history) == 200
    assert session.history[-1]["source"] == "human"


def test_training_opponent_sampling_and_action_routes(tmp_path, monkeypatch):
    from qwixx_ai import distributed
    from qwixx_ai.core import Game

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    coord = distributed.Coordinator()
    try:
        sentinel_best = object()
        sentinel_league = object()
        coord.best_model = sentinel_best
        coord.league.clear()
        coord.league.append(sentinel_league)

        for roll, expected in (
            (0.10, ("best", sentinel_best)),
            (0.50, ("league", sentinel_league)),
            (0.80, ("rollout", None)),
            (0.99, ("heuristic", None)),
        ):
            monkeypatch.setattr(coord.rng, "random", lambda value=roll: value)
            monkeypatch.setattr(coord.rng, "choice", lambda values: values[0])
            assert coord._sample_training_opponent() == expected

        game = Game(seed=101)
        monkeypatch.setattr(distributed, "choose", lambda model, game, epsilon=0: ("choose", epsilon))
        monkeypatch.setattr(distributed, "rollout_policy", lambda game: ("rollout", 0))
        monkeypatch.setattr(distributed, "heuristic_v2", lambda game: ("heuristic", 0))

        current = coord._training_action(("current", object()), game)
        assert current[0] == "choose"
        assert current[1] == coord.exploration()
        assert coord._training_action(("best", object()), game) == ("choose", 0.04)
        assert coord._training_action(("league", object()), game) == ("choose", 0.04)
        assert coord._training_action(("rollout", None), game) == ("rollout", 0)
        assert coord._training_action(("heuristic", None), game) == ("heuristic", 0)
    finally:
        coord.db.close()


def test_ticket_cleanup_limit_history_and_stats(tmp_path, monkeypatch):
    import time
    from qwixx_ai import distributed

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    coord = distributed.Coordinator()
    try:
        coord.pending["old"] = (1, time.time() - 901)
        ticket = coord.ticket()
        assert "old" not in coord.pending
        assert ticket["policy_version"] == coord.version

        monkeypatch.setattr(distributed, "MAX_PENDING", 1)
        coord.pending = {"full": (1, time.time())}
        with pytest.raises(ValueError, match="Too many outstanding tickets"):
            coord.ticket()

        with coord.lock:
            coord._record("browser", [1, 2, 3, 4], None, 7)
        assert len(coord.history(0)) == 1
        assert len(coord.history(9999)) == 1
        stats = coord.stats()
        assert stats["database_games"] == 1
        assert stats["sessions"] == 0
        assert stats["league_size"] == len(coord.league)
    finally:
        coord.db.close()


def test_accept_rejects_bad_codes_and_incomplete_game(tmp_path, monkeypatch):
    from qwixx_ai import distributed

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    coord = distributed.Coordinator()
    try:
        for moves, message in (
            ([89], "Invalid move code"),
            ([-1], "Invalid move code"),
            ([1.5], "Invalid move code"),
            ([], "Incomplete game"),
        ):
            ticket = coord.ticket()
            with pytest.raises(ValueError, match=message):
                coord.accept({"ticket": ticket["ticket"], "moves": moves})
    finally:
        coord.db.close()


def test_checkpoint_skips_recent_live_session(tmp_path, monkeypatch):
    import time
    from qwixx_ai import distributed
    from qwixx_ai.web import Session

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    monkeypatch.setenv("QWIXX_CHECKPOINT_IDLE_SECONDS", "120")
    coord = distributed.Coordinator()
    try:
        coord.sessions["live"] = (Session(difficulty="intermediate"), time.time(), None)
        assert coord.checkpoint(force=False) is False
    finally:
        coord.db.close()


def test_new_session_expires_stale_sessions_and_checks_player(tmp_path, monkeypatch):
    import time
    from qwixx_ai import distributed

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    coord = distributed.Coordinator()
    try:
        coord.sessions["stale"] = (object(), time.time() - 90000, None)
        seen = []
        monkeypatch.setattr(coord.player_stats, "profile", lambda player_id: seen.append(player_id) or {"id": player_id})
        result = coord.new_session("intermediate", player_id="player-1", randomize_seating=False)
        assert "stale" not in coord.sessions
        assert seen == ["player-1"]
        assert result["session_id"] in coord.sessions
    finally:
        coord.db.close()


def test_learn_executes_optimizer_and_updates_priorities(tmp_path, monkeypatch):
    from qwixx_ai import distributed
    from qwixx_ai.core import Game
    from qwixx_ai.env import observe, mask

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    coord = distributed.Coordinator()
    try:
        game = Game(seed=111)
        state = observe(game).copy()
        next_state = observe(game).copy()
        next_mask = mask(game).copy()
        sample = (state, 0, 0.25, next_state, False, next_mask)
        coord._add_transitions([sample] * 300)
        before = list(coord.priorities)
        coord.learn()
        assert coord.updates == 1
        assert coord.last_loss is not None
        assert any(a != b for a, b in zip(before, coord.priorities))
    finally:
        coord.db.close()


def test_forced_checkpoint_evaluates_existing_champion_without_promotion(tmp_path, monkeypatch):
    import copy
    from qwixx_ai import distributed

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    coord = distributed.Coordinator()
    try:
        coord.best_model = copy.deepcopy(coord.model).eval()
        coord.best = 0.50
        coord.league.clear()
        monkeypatch.setattr(
            distributed,
            "evaluate",
            lambda model, games, seed: {"win_share": 0.50, "mean_score": 60.0},
        )
        old_version = coord.version
        assert coord.checkpoint(force=True) is True
        assert coord.version == old_version + 1
        assert coord.best == 0.50
        assert coord.last_eval["champion_win_share"] == 0.50
        assert len(coord.league) == 0
        assert (tmp_path / "latest.pt").exists()
        assert (tmp_path / "trainer.pt").exists()
        assert coord.log.exists()
    finally:
        coord.db.close()


def test_authenticated_http_stats_and_auth_routes(tmp_path, monkeypatch):
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from qwixx_ai import distributed

    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    coord = distributed.Coordinator()
    monkeypatch.setattr(coord.player_stats, "authenticate", lambda token: "player-1")
    monkeypatch.setattr(coord.player_stats, "profile", lambda pid: {"id": pid, "name": "Dwight"})
    monkeypatch.setattr(coord.player_stats, "stats", lambda pid, difficulty: {"player_id": pid, "difficulty": difficulty})
    monkeypatch.setattr(coord.player_stats, "ai_stats", lambda difficulty: {"difficulty": difficulty, "games": 3})
    monkeypatch.setattr(coord.player_stats, "leaderboard", lambda difficulty: {"difficulty": difficulty, "leaders": []})
    monkeypatch.setattr(coord.player_stats, "request_code", lambda email, name: {"sent": True, "email": email, "name": name})
    monkeypatch.setattr(
        coord.player_stats,
        "verify_code",
        lambda email, code: {"token": "session-token", "player": {"id": "player-1", "email": email}},
    )
    logged_out = []
    monkeypatch.setattr(coord.player_stats, "logout", lambda token: logged_out.append(token))

    class TestHandler(distributed.Handler):
        def log_message(self, *args):
            pass

    TestHandler.coord = coord
    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        req = urllib.request.Request(base + path, headers={"Cookie": "qwixx_auth=abc"})
        return urllib.request.urlopen(req, timeout=5)

    def post(path, payload):
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Cookie": "qwixx_auth=abc"},
        )
        return urllib.request.urlopen(req, timeout=5)

    try:
        with get("/api/auth/me") as response:
            assert json.load(response)["player"]["id"] == "player-1"
        with get("/api/player-stats?difficulty=champion") as response:
            assert json.load(response)["difficulty"] == "champion"
        with get("/api/ai-stats?difficulty=advanced") as response:
            assert json.load(response)["games"] == 3
        with get("/api/leaderboard?difficulty=champion") as response:
            assert json.load(response)["leaders"] == []

        with post("/api/auth/request", {"email": "d@example.com", "name": "Dwight"}) as response:
            assert json.load(response)["sent"] is True
        with post("/api/auth/verify", {"email": "d@example.com", "code": "123456"}) as response:
            body = json.load(response)
            assert body["player"]["id"] == "player-1"
            assert "qwixx_auth=session-token" in response.headers["Set-Cookie"]
        with post("/api/auth/logout", {}) as response:
            assert response.status == 200
        assert logged_out == ["abc"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        coord.db.close()
