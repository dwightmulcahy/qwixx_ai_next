"""Centralized training coordinator with browser-generated, server-verified games."""
import json, os, random, secrets, threading, time, copy, uuid, sqlite3, smtplib, logging
from collections import deque
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch
from torch import nn
from .core import Game
from .env import observe,mask,encode,decode
from .model import network,choose,save,load
from .agents import heuristic,random_agent
from .tournament import evaluate
from .web import HTML as GAME_HTML, Session
from .player_stats import PlayerStats

ROOT=Path(os.getenv('QWIXX_DATA', Path.cwd()/'runs'))
ROOT.mkdir(parents=True,exist_ok=True)
WORKER_TOKEN=os.getenv('QWIXX_WORKER_TOKEN','')
ADMIN_TOKEN=os.getenv('QWIXX_ADMIN_TOKEN','')
MAX_BODY=350_000
MAX_PENDING=500


class Coordinator:
    def __init__(self):
        self.lock=threading.RLock();self.rng=random.Random()
        self.pending={};self.replay=deque(maxlen=50000);self.priorities=deque(maxlen=50000)
        self.db=sqlite3.connect(ROOT/'training.sqlite3',check_same_thread=False,timeout=20)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS games (id INTEGER PRIMARY KEY, source TEXT NOT NULL, scores TEXT NOT NULL, moves TEXT, created REAL NOT NULL, seed INTEGER)')
        columns={row[1] for row in self.db.execute('PRAGMA table_info(games)')}
        if 'seed' not in columns:self.db.execute('ALTER TABLE games ADD COLUMN seed INTEGER')
        self.db.commit()
        self.recent_games=deque(maxlen=200)
        self.started_at=time.time()
        self._restore_replay()
        self.games=0;self.browser_games=0;self.central_games=0;self.updates=0
        self.best=-1.;self.version=0;self.last_loss=None;self.last_eval=None
        self.player_stats=PlayerStats(ROOT/'players.sqlite3')
        self.sessions={};self.promoted_at=None;self.best_model=None;self.league=deque(maxlen=6)
        self.obs_size=len(observe(Game()))
        self.model=network(self.obs_size);self.target=network(self.obs_size)
        checkpoint=ROOT/'latest.pt'
        if checkpoint.exists():
            try:self.model=load(checkpoint)
            except Exception as exc:print('Checkpoint load failed:',exc,flush=True)
        self.target.load_state_dict(self.model.state_dict())
        best_path=ROOT/'best.pt'
        if best_path.exists():
            try:
                self.best_model=load(best_path)
                self.best=float(torch.load(best_path,map_location='cpu',weights_only=True).get('win_share',-1))
            except Exception as exc:print('Best checkpoint load failed:',exc,flush=True)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=3e-4)
        state_path=ROOT/'trainer.pt'
        if state_path.exists():
            try:
                state=torch.load(state_path,map_location='cpu',weights_only=True)
                self.optimizer.load_state_dict(state['optimizer'])
                self.updates=state.get('updates',0)
                self.games=state.get('games',0)
                self.version=state.get('version',0)
            except Exception as exc:print('Trainer state recovery failed:',exc,flush=True)
        self.stop=threading.Event();self.paused=threading.Event()
        self.log=ROOT/'metrics.jsonl'
    def ticket(self):
        with self.lock:
            now=time.time()
            for key,(_,issued) in list(self.pending.items()):
                if now-issued>900:self.pending.pop(key,None)
            if len(self.pending)>=MAX_PENDING:raise ValueError('Too many outstanding tickets')
            token=secrets.token_urlsafe(24);seed=self.rng.randrange(2**31)
            self.pending[token]=(seed,time.time())
            return {'ticket':token,'seed':seed,'max_moves':1000,'policy_version':self.version}
    def accept(self,data):
        token=data.get('ticket');moves=data.get('moves')
        if not isinstance(token,str) or not isinstance(moves,list) or len(moves)>1000:raise ValueError('Invalid submission')
        with self.lock:
            item=self.pending.pop(token,None)
            if item is None or time.time()-item[1]>900:raise ValueError('Expired or unknown ticket')
            seed=item[0]
        game=Game(seed=seed);last=[None]*4;transitions=[]
        try:
            for code in moves:
                if type(code)!=int or not 0<=code<89:raise ValueError('Invalid move code')
                if game.done:raise ValueError('Extra moves after game end')
                if not mask(game)[code]:raise ValueError('Illegal move')
                who=game.actor
                if last[who] is not None:
                    s,a,start=last[who]
                    transitions.append((s,a,(game.scores()[who]-start)/25,observe(game),False,mask(game)))
                last[who]=(observe(game).copy(),code,game.scores()[who]);game.step(decode(code))
            if not game.done:raise ValueError('Incomplete game')
        except Exception as exc:raise ValueError(str(exc)) from exc
        scores=game.scores()
        for who,item in enumerate(last):
            if item is not None:
                s,a,start=item
                reward=(scores[who]-start)/25+(1 if scores[who]==max(scores) else -1)
                transitions.append((s,a,reward,observe(game),True,mask(game)))
        with self.lock:
            self._add_transitions(transitions);self.games+=1;self.browser_games+=1
            self._record('browser',scores,moves,seed)
        return {'accepted':True,'scores':scores,'transitions':len(transitions)}
    def central_rollout(self):
        seed=self.rng.randrange(2**31)
        g=Game(seed=seed);last=[None]*4;transitions=[];moves=[]
        while not g.done:
            who=g.actor
            if last[who] is not None:
                s,a,start=last[who]
                transitions.append((s,a,(g.scores()[who]-start)/25,observe(g),False,mask(g)))
            with self.lock:
                policy=self.rng.choice(list(self.league)) if self.league and self.rng.random()<.15 else (self.best_model if self.best_model is not None and self.rng.random()<.35 else self.model)
                a=choose(policy,g,epsilon=self.exploration()) if self.rng.random()<.65 else (heuristic(g) if self.rng.random()<.8 else random_agent(g))
            last[who]=(observe(g).copy(),encode(a),g.scores()[who]);moves.append(encode(a));g.step(a)
        scores=g.scores()
        for who,item in enumerate(last):
            if item is not None:
                s,a,start=item
                transitions.append((s,a,(scores[who]-start)/25+(1 if scores[who]==max(scores) else -1),observe(g),True,mask(g)))
        with self.lock:
            self._add_transitions(transitions);self.games+=1;self.central_games+=1
            self._record('central',scores,moves,seed)
    def exploration(self):
        return max(0.04,0.30*(0.99997**self.updates))
    def _add_transitions(self,items):
        self.replay.extend(items)
        self.priorities.extend([max(self.priorities,default=1.0)]*len(items))
    def _record(self,source,scores,moves,seed=None):
        self.db.execute('INSERT INTO games(source,scores,moves,created,seed) VALUES(?,?,?,?,?)',(source,json.dumps(scores),json.dumps(moves) if moves is not None else None,time.time(),seed))
        self.db.commit()
        self.recent_games.append({'source':source,'scores':scores})
    def _restore_replay(self):
        # Reconstruct trajectories from server-validated deterministic game logs.
        rows=self.db.execute('SELECT seed,moves FROM games WHERE seed IS NOT NULL AND moves IS NOT NULL ORDER BY id DESC LIMIT 100').fetchall()
        for seed,encoded in reversed(rows):
            try:
                g=Game(seed=seed);last=[None]*g.n;transitions=[]
                for code in json.loads(encoded):
                    if g.done or not mask(g)[code]:raise ValueError('Incompatible game log')
                    who=g.actor
                    if last[who] is not None:
                        state,action,start=last[who]
                        transitions.append((state,action,(g.scores()[who]-start)/25,observe(g),False,mask(g)))
                    last[who]=(observe(g).copy(),code,g.scores()[who]);g.step(decode(code))
                if not g.done:continue
                scores=g.scores()
                for who,item in enumerate(last):
                    if item is not None:
                        state,action,start=item
                        transitions.append((state,action,(scores[who]-start)/25+(1 if scores[who]==max(scores) else -1),observe(g),True,mask(g)))
                self._add_transitions(transitions)
            except (ValueError,TypeError,IndexError):continue
    def history(self,limit=100):
        with self.lock:
            rows=self.db.execute('SELECT source,scores,created FROM games ORDER BY id DESC LIMIT ?',(min(max(int(limit),1),500),)).fetchall()
            return [{'source':src,'scores':json.loads(scores),'time':created} for src,scores,created in rows]
    def learn(self):
        with self.lock:
            if len(self.replay)<256:return
            # Proportional TD-error prioritized experience replay.
            population=list(self.replay)
            weights=np.asarray(self.priorities,dtype=np.float64)**0.6
            probabilities=weights/weights.sum()
            indices=self.rng.choices(range(len(population)),weights=probabilities,k=64)
            batch=[population[i] for i in indices]
            ss,aa,rr,ns,dd,mm=zip(*batch)
            q=self.model(torch.from_numpy(np.stack(ss))).gather(1,torch.tensor(aa).long()[:,None]).squeeze(1)
            with torch.no_grad():
                nextq=self.target(torch.from_numpy(np.stack(ns)))
                nextq[~torch.from_numpy(np.stack(mm))]=-1e9
                # Double DQN: select with online model, evaluate with target model.
                online_next=self.model(torch.from_numpy(np.stack(ns)))
                online_next[~torch.from_numpy(np.stack(mm))]=-1e9
                best_actions=online_next.argmax(1,keepdim=True)
                target=torch.tensor(rr,dtype=torch.float32)+.99*(1-torch.tensor(dd,dtype=torch.float32))*nextq.gather(1,best_actions).squeeze(1)
            errors=(q-target).detach().abs().cpu().numpy()
            importance=(len(population)*probabilities[indices])**(-0.4)
            importance=importance/importance.max()
            losses=nn.functional.smooth_l1_loss(q,target,reduction='none')
            loss=(losses*torch.tensor(importance,dtype=torch.float32)).mean()
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),5);self.optimizer.step()
            for index,error in zip(indices,errors):self.priorities[index]=float(error)+0.01
            self.updates+=1;self.last_loss=round(float(loss.item()),5)
            if self.updates%300==0:self.target.load_state_dict(self.model.state_dict())
    def checkpoint(self):
        # Hold the training lock while evaluating/saving so model weights remain stable.
        with self.lock:
            temp=ROOT/'latest.tmp.pt'
            save(self.model,temp,updates=self.updates,games=self.games)
            temp.replace(ROOT/'latest.pt')
            state_tmp=ROOT/'trainer.tmp.pt'
            torch.save({'optimizer':self.optimizer.state_dict(),'updates':self.updates,'games':self.games,'version':self.version},state_tmp)
            state_tmp.replace(ROOT/'trainer.pt')
            self.version+=1
            result=evaluate(self.model,192,seed=123450)
            self.last_eval=result
            # Promotion requires an improvement on fixed evaluation seeds.
            # Candidate must outperform champion under identical evaluation seeds.
            champion_eval=evaluate(self.best_model,192,seed=123450) if self.best_model is not None else None
            result['champion_win_share']=champion_eval['win_share'] if champion_eval else None
            if champion_eval is None or result['win_share']>champion_eval['win_share']+0.03:
                candidate=ROOT/'candidate.pt'
                save(self.model,candidate,updates=self.updates,games=self.games,win_share=result['win_share'])
                candidate.replace(ROOT/'best.pt')
                self.best=result['win_share']
                self.best_model=copy.deepcopy(self.model).eval()
                self.league.append(copy.deepcopy(self.best_model))
                self.promoted_at=time.time()
            with self.log.open('a') as f:f.write(json.dumps({'time':time.time(),**self.stats()})+'\n')
    def new_session(self,difficulty='champion',player_id=None):
        with self.lock:
            for key,(_,updated,_) in list(self.sessions.items()):
                if time.time()-updated>86400:self.sessions.pop(key,None)
            if len(self.sessions)>=200:raise ValueError('Too many game sessions')
            if player_id is not None:self.player_stats.profile(player_id)
            sid=uuid.uuid4().hex
            model=self.best_model
            session=Session(model=model,difficulty=difficulty)
            session.advance()
            self.sessions[sid]=(session,time.time(),player_id)
            return {'session_id':sid,**session.view()}
    def session_action(self,sid,action,expected_turn=None,expected_phase=None):
        with self.lock:
            entry=self.sessions.get(sid)
            if not entry:raise ValueError('Session expired; start a new game')
            session,_,player_id=entry
            with session.lock:
                if expected_turn is not None and (type(expected_turn) is not int or expected_turn!=session.game.turn):
                    raise ValueError('Game state changed; start a new game or reload')
                if expected_phase is not None and (type(expected_phase) is not int or expected_phase!=session.game.phase):
                    raise ValueError('Step already advanced; reload game')
                session.apply_action(action,'human')
                session.advance()
                view=session.view()
                if session.game.done and player_id is not None:
                    self.player_stats.record(sid,player_id,session.difficulty,session.game)
            self.sessions[sid]=(session,time.time(),player_id)
            return {'session_id':sid,**view}
    def policy_export(self):
        # Export the promoted model for local JS inference in browser workers.
        with self.lock:
            model=self.best_model or self.model
            layers=[]
            for layer in model:
                if isinstance(layer,nn.Linear):
                    layers.append({'w':layer.weight.detach().tolist(),'b':layer.bias.detach().tolist()})
            return {'version':self.version,'layers':layers,'observation_size':self.obs_size}
    def stats(self):
        return {'paused':self.paused.is_set(),'uptime_seconds':int(time.time()-self.started_at),'database_games':self.db.execute('SELECT COUNT(*) FROM games').fetchone()[0],'games':self.games,'browser_games':self.browser_games,'central_games':self.central_games,'replay_size':len(self.replay),'updates':self.updates,'model_version':self.version,'loss':self.last_loss,'epsilon':self.exploration(),'best_win_share':self.best,'evaluation':self.last_eval,'promoted_at':self.promoted_at,'sessions':len(self.sessions),'league_size':len(self.league)}
    def run_rollouts(self):
        while not self.stop.is_set():
            if self.paused.is_set():
                self.stop.wait(.1);continue
            try:self.central_rollout()
            except Exception as exc:
                print('Rollout error:',repr(exc),flush=True);self.stop.wait(2)
    def run(self):
        last=time.monotonic()
        while not self.stop.is_set():
            if self.paused.is_set():
                self.stop.wait(.1);continue
            try:
                for _ in range(3):self.learn()
                if time.monotonic()-last>60:
                    self.checkpoint();last=time.monotonic()
                self.stop.wait(.05)
            except Exception as exc:
                print('Coordinator error:',repr(exc),flush=True);self.stop.wait(2)

WORKER_JS=r'''
let enabled=false,completed=0,busy=false,policy=null;
const inWorker=typeof document==='undefined';
const order=[Array.from({length:11},(_,i)=>i+2),Array.from({length:11},(_,i)=>i+2),Array.from({length:11},(_,i)=>12-i),Array.from({length:11},(_,i)=>12-i)];
async function refreshPolicy(){const r=await fetch('/api/policy');if(!r.ok)throw Error('policy');policy=await r.json()}
function dot(w,x,b){let y=b;for(let i=0;i<x.length;i++)y+=w[i]*x[i];return y}
function infer(obs){let x=obs;for(let l=0;l<policy.layers.length;l++){const p=policy.layers[l],y=p.w.map((w,i)=>dot(w,x,p.b[i]));x=l<policy.layers.length-1?y.map(v=>Math.max(0,v)):y}return x}
function legal(board,c,n,locks){const row=board.rows[c];if(locks.includes(c))return false;const idx=order[c].indexOf(n);if(idx<0||row.includes(n))return false;if(row.some(x=>order[c].indexOf(x)>=idx))return false;if(idx===10&&row.length<5)return false;return true}
function maskState(s){const out=Array(89).fill(false);out[88]=true;const sums=s.phase===s.boards.length?s.dice.slice(0,2).flatMap(w=>s.dice.slice(2).map((c,i)=>[i,w+c])):[[null,s.dice[0]+s.dice[1]]];for(const [c,n] of sums){for(let color=0;color<4;color++){if(c!==null&&c!==color)continue;if(legal(s.boards[s.active],color,n,s.locks))out[color*22+(n-2)]=true}}return out}
function encodeState(s){const out=[];for(const b of s.boards){for(let c=0;c<4;c++){for(const n of order[c])out.push(b.rows[c].includes(n)?1:0)}out.push(b.penalties/4)}for(const d of s.dice)out.push(d/6);out.push(s.active/4,s.phase/5,s.turn/100);return out}
async function play(){if(busy||!enabled)return;busy=true;try{if(!policy)await refreshPolicy();const j=await (await fetch('/api/job')).json();let s={boards:Array.from({length:4},()=>({rows:[[],[],[],[]],penalties:0})),locks:[],active:0,phase:0,turn:0,dice:j.rolls[0]},moves=[];for(let step=0;step<j.max_moves;step++){const m=maskState(s),q=infer(encodeState(s));let a=88,b=-Infinity;for(let i=0;i<89;i++)if(m[i]&&q[i]>b){b=q[i];a=i}moves.push(a);const rr=await (await fetch('/api/sim',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({state:s,action:a,rolls:j.rolls})})).json();s=rr.state;if(rr.done)break}const r=await fetch('/api/result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticket:j.ticket,moves})});if(r.ok)completed++;else console.warn('result rejected',await r.text())}catch(e){console.warn('worker error',e);await new Promise(r=>setTimeout(r,2000))}finally{busy=false;if(enabled)setTimeout(play,0)}}
function start(){enabled=true;play()}function stop(){enabled=false}
if(inWorker)self.onmessage=e=>{if(e.data==='start')start();if(e.data==='stop')stop();if(e.data==='count')postMessage(completed)};
'''

class Handler(BaseHTTPRequestHandler):
    coord=None;session=None
    rate_lock=threading.Lock();rate_buckets={}
    def limited(self):
        now=time.monotonic();ip=self.client_address[0]
        with self.rate_lock:
            bucket=[t for t in self.rate_buckets.get(ip,[]) if now-t<60]
            limited=len(bucket)>=120
            if not limited:bucket.append(now)
            self.rate_buckets[ip]=bucket
        return limited
    def reply(self,obj,status=200,ctype='application/json'):
        b=(json.dumps(obj) if ctype=='application/json' else obj).encode();self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def authorized_worker(self):
        return not WORKER_TOKEN or secrets.compare_digest(self.headers.get('X-Worker-Token',''),WORKER_TOKEN)
    def auth_token(self):
        from http.cookies import SimpleCookie
        cookie=SimpleCookie()
        try:cookie.load(self.headers.get('Cookie',''))
        except Exception:return ''
        return cookie['qwixx_auth'].value if 'qwixx_auth' in cookie else ''
    def player_id(self):
        return self.coord.player_stats.authenticate(self.auth_token())
    def do_GET(self):
        try:
            if self.path=='/':return self.reply(DASHBOARD_HTML,ctype='text/html; charset=utf-8')
            if self.path=='/worker.js':return self.reply(WORKER_JS,ctype='application/javascript; charset=utf-8')
            if self.path=='/health':return self.reply({'healthy':True})
            if self.path=='/api/stats':return self.reply(self.coord.stats())
            if self.path.startswith('/api/history'):
                try:limit=int(self.path.split('limit=',1)[1].split('&',1)[0]) if 'limit=' in self.path else 100
                except ValueError:limit=100
                return self.reply({'games':self.coord.history(limit)})
            if self.path=='/api/auth/me':
                pid=self.player_id()
                return self.reply({'player':self.coord.player_stats.profile(pid) if pid else None})
            if self.path.startswith('/api/player-stats?'):
                from urllib.parse import parse_qs, urlsplit
                query=parse_qs(urlsplit(self.path).query)
                pid=self.player_id()
                if not pid:return self.reply({'error':'Sign in required'},401)
                return self.reply(self.coord.player_stats.stats(pid,query.get('difficulty',[None])[0]))
            if self.path=='/api/policy':return self.reply(self.coord.policy_export())
            if self.path=='/api/job':
                if self.limited():return self.reply({'error':'Rate limit exceeded'},429)
                if not self.authorized_worker():return self.reply({'error':'Unauthorized worker'},403)
                job=self.coord.ticket();job['rolls']=rolls_for(job['seed']);return self.reply(job)
            self.send_error(404)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:self.reply({'error':str(e)},500)
    def do_POST(self):
        if self.path=='/api/admin':
            if not ADMIN_TOKEN or not secrets.compare_digest(self.headers.get('X-Admin-Token',''),ADMIN_TOKEN):return self.reply({'error':'Unauthorized'},403)
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<1 or n>1000:raise ValueError('Invalid body size')
                command=json.loads(self.rfile.read(n)).get('command')
                if command=='pause':self.coord.paused.set()
                elif command=='resume':self.coord.paused.clear()
                elif command=='checkpoint':self.coord.checkpoint()
                elif command=='rollback':
                    with self.coord.lock:
                        if self.coord.best_model is None:raise ValueError('No champion checkpoint')
                        self.coord.model.load_state_dict(self.coord.best_model.state_dict())
                        self.coord.target.load_state_dict(self.coord.best_model.state_dict())
                        self.coord.optimizer=torch.optim.Adam(self.coord.model.parameters(),lr=3e-4)
                        self.coord.checkpoint()
                else:raise ValueError('Unknown command')
                return self.reply({'ok':True,'paused':self.coord.paused.is_set()})
            except (ValueError,TypeError) as exc:return self.reply({'error':str(exc)},400)
        if self.path in ('/api/auth/request','/api/auth/verify','/api/auth/logout'):
            try:
                if self.limited():return self.reply({'error':'Too many requests'},429)
                n=int(self.headers.get('Content-Length','0'))
                if n<0 or n>1024:raise ValueError('Invalid body size')
                data=json.loads(self.rfile.read(n) or b'{}')
                if self.path=='/api/auth/request':return self.reply(self.coord.player_stats.request_code(data.get('email'),data.get('name')))
                if self.path=='/api/auth/logout':
                    self.coord.player_stats.logout(self.auth_token())
                    self.send_response(200);self.send_header('Set-Cookie','qwixx_auth=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0');self.send_header('Content-Length','2');self.end_headers();self.wfile.write(b'{}');return
                result=self.coord.player_stats.verify_code(data.get('email'),data.get('code'))
                cookie='qwixx_auth='+result['token']+'; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000'
                if self.headers.get('X-Forwarded-Proto','').lower()=='https' or os.getenv('QWIXX_COOKIE_SECURE','false').lower()=='true':cookie+='; Secure'
                body=json.dumps({'player':result['player']}).encode()
                self.send_response(200);self.send_header('Set-Cookie',cookie);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            except (ValueError,TypeError,RuntimeError) as e:return self.reply({'error':str(e)},400 if isinstance(e,ValueError) else 503)
            except (smtplib.SMTPException,OSError):
                logging.exception('Email verification SMTP delivery failed')
                return self.reply({'error':'Email delivery failed. Check Gmail app password and SMTP connectivity.'},503)
        if self.path in ('/reset','/move'):
            try:
                n=int(self.headers.get('Content-Length','0'));data=json.loads(self.rfile.read(n) or b'{}')
                if self.path=='/reset':result=self.coord.new_session(data.get('difficulty','champion'),self.player_id())
                else:
                    a=data.get('action');a=None if a is None else tuple(a)
                    result=self.coord.session_action(data.get('session_id'),a,data.get('expected_turn'),data.get('expected_phase'))
                return self.reply(result)
            except (ValueError,TypeError) as e:return self.reply({'error':str(e)},400)
        if self.path=='/api/result':
            if self.limited():return self.reply({'error':'Rate limit exceeded'},429)
            if not self.authorized_worker():return self.reply({'error':'Unauthorized worker'},403)
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<2 or n>MAX_BODY:raise ValueError('Invalid body')
                return self.reply(self.coord.accept(json.loads(self.rfile.read(n))))
            except (ValueError,TypeError,json.JSONDecodeError) as e:return self.reply({'error':str(e)},400)
        if self.path=='/api/sim':
            if self.limited():return self.reply({'error':'Rate limit exceeded'},429)
            if not self.authorized_worker():return self.reply({'error':'Unauthorized worker'},403)
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<2 or n>MAX_BODY:raise ValueError('Invalid body')
                data=json.loads(self.rfile.read(n));s=data['state'];a=data['action'];rolls=data['rolls']
                g=Game(seed=0);g.active=s['active'];g.phase=s['phase'];g.turn=s['turn'];g.dice=tuple(s['dice']);g.locks=set(s['locks'])
                for i,b in enumerate(s['boards']):g.boards[i].rows=[list(r) for r in b['rows']];g.boards[i].penalties=b['penalties']
                if not mask(g)[a]:raise ValueError('Illegal move')
                g.step(decode(a))
                if not g.done and g.phase==0:g.dice=tuple(rolls[g.turn])
                return self.reply({'state':{'boards':[{'rows':b.rows,'penalties':b.penalties} for b in g.boards],'locks':list(g.locks),'active':g.active,'phase':g.phase,'turn':g.turn,'dice':g.dice},'done':g.done})
            except (ValueError,TypeError,KeyError,IndexError,json.JSONDecodeError) as e:return self.reply({'error':str(e)},400)
        self.send_error(404)

def rolls_for(seed):
    r=random.Random(seed);return [[r.randint(1,6) for _ in range(6)] for _ in range(1000)]

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    c=Coordinator();Handler.coord=c
    threading.Thread(target=c.run,daemon=True).start()
    for _ in range(max(0,int(os.getenv('QWIXX_CENTRAL_WORKERS','0')))):threading.Thread(target=c.run_rollouts,daemon=True).start()
    server=ThreadingHTTPServer((a.host,a.port),Handler)
    print(f'Qwixx distributed server: http://{a.host}:{a.port} | persistent data: {ROOT}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:c.stop.set();server.server_close();c.checkpoint()

if __name__=='__main__':main()
