"""Centralized training coordinator with browser-generated, server-verified games."""
import json, os, random, secrets, threading, time, copy, uuid, sqlite3, tempfile, signal
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

ROOT=Path(os.getenv('QWIXX_DATA','/app/runs'))
ROOT.mkdir(parents=True,exist_ok=True)
WORKER_TOKEN=os.getenv('QWIXX_WORKER_TOKEN','')
MAX_BODY=350_000
MAX_PENDING=500


class Coordinator:
    def __init__(self):
        self.lock=threading.RLock();self.rng=random.Random()
        self.pending={};self.replay=deque(maxlen=50000)
        self.db=sqlite3.connect(ROOT/'training.sqlite3',check_same_thread=False,timeout=20)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS games (id INTEGER PRIMARY KEY, source TEXT NOT NULL, scores TEXT NOT NULL, moves TEXT, created REAL NOT NULL)')
        self.db.commit()
        self.recent_games=deque(maxlen=200)
        self.started_at=time.time()
        self.games=0;self.browser_games=0;self.central_games=0;self.updates=0
        self.best=-1.;self.version=0;self.last_loss=None;self.last_eval=None
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
        self.stop=threading.Event()
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
            self.replay.extend(transitions);self.games+=1;self.browser_games+=1
            self._record('browser',scores,moves)
        return {'accepted':True,'scores':scores,'transitions':len(transitions)}
    def central_rollout(self):
        g=Game(seed=self.rng.randrange(2**31));last=[None]*4;transitions=[]
        while not g.done:
            who=g.actor
            if last[who] is not None:
                s,a,start=last[who]
                transitions.append((s,a,(g.scores()[who]-start)/25,observe(g),False,mask(g)))
            with self.lock:
                policy=self.rng.choice(list(self.league)) if self.league and self.rng.random()<.15 else (self.best_model if self.best_model is not None and self.rng.random()<.35 else self.model)
                a=choose(policy,g,epsilon=.18) if self.rng.random()<.65 else (heuristic(g) if self.rng.random()<.8 else random_agent(g))
            last[who]=(observe(g).copy(),encode(a),g.scores()[who]);g.step(a)
        scores=g.scores()
        for who,item in enumerate(last):
            if item is not None:
                s,a,start=item
                transitions.append((s,a,(scores[who]-start)/25+(1 if scores[who]==max(scores) else -1),observe(g),True,mask(g)))
        with self.lock:
            self.replay.extend(transitions);self.games+=1;self.central_games+=1
            self._record('central',scores,None)
    def _record(self,source,scores,moves):
        self.db.execute('INSERT INTO games(source,scores,moves,created) VALUES(?,?,?,?)',(source,json.dumps(scores),json.dumps(moves) if moves is not None else None,time.time()))
        self.db.commit()
        self.recent_games.append({'source':source,'scores':scores})
    def history(self,limit=100):
        with self.lock:
            rows=self.db.execute('SELECT source,scores,created FROM games ORDER BY id DESC LIMIT ?',(min(max(int(limit),1),500),)).fetchall()
            return [{'source':src,'scores':json.loads(scores),'time':created} for src,scores,created in rows]
    def learn(self):
        with self.lock:
            if len(self.replay)<256:return
            batch=self.rng.sample(list(self.replay),64)
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
            loss=nn.functional.smooth_l1_loss(q,target)
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),5);self.optimizer.step()
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
            if result['win_share']>self.best+0.001:
                candidate=ROOT/'candidate.pt'
                save(self.model,candidate,updates=self.updates,games=self.games,win_share=result['win_share'])
                candidate.replace(ROOT/'best.pt')
                self.best=result['win_share']
                self.best_model=copy.deepcopy(self.model).eval()
                self.league.append(copy.deepcopy(self.best_model))
                self.promoted_at=time.time()
            with self.log.open('a') as f:f.write(json.dumps({'time':time.time(),**self.stats()})+'\n')
    def new_session(self):
        with self.lock:
            for key,(_,updated) in list(self.sessions.items()):
                if time.time()-updated>86400:self.sessions.pop(key,None)
            if len(self.sessions)>=200:raise ValueError('Too many game sessions')
            sid=uuid.uuid4().hex
            model=self.best_model
            session=Session(model=model)
            session.advance()
            self.sessions[sid]=(session,time.time())
            return {'session_id':sid,**session.view()}
    def session_action(self,sid,action):
        with self.lock:
            entry=self.sessions.get(sid)
            if not entry:raise ValueError('Session expired; start a new game')
            session,_=entry
            with session.lock:
                session.game.step(action)
                session.advance()
                view=session.view()
            self.sessions[sid]=(session,time.time())
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
        return {'uptime_seconds':int(time.time()-self.started_at),'database_games':self.db.execute('SELECT COUNT(*) FROM games').fetchone()[0],'games':self.games,'browser_games':self.browser_games,'central_games':self.central_games,'replay_size':len(self.replay),'updates':self.updates,'model_version':self.version,'loss':self.last_loss,'best_win_share':self.best,'evaluation':self.last_eval,'promoted_at':self.promoted_at,'sessions':len(self.sessions),'league_size':len(self.league)}
    def run(self):
        last=time.monotonic();cycles=0
        while not self.stop.is_set():
            try:
                self.central_rollout()
                for _ in range(3):self.learn()
                cycles+=1
                if time.monotonic()-last>60:
                    self.checkpoint();last=time.monotonic()
                time.sleep(.02)
            except Exception as exc:
                print('Coordinator error:',repr(exc),flush=True);time.sleep(2)

WORKER_JS=r'''
let enabled=false,completed=0,busy=false,policy=null;
const inWorker=typeof document==='undefined';
const order=[Array.from({length:11},(_,i)=>i+2),Array.from({length:11},(_,i)=>i+2),Array.from({length:11},(_,i)=>12-i),Array.from({length:11},(_,i)=>12-i)];
async function refreshPolicy(){const r=await fetch('/api/policy');if(!r.ok)throw Error('Policy unavailable');policy=await r.json();}
function inference(input){let v=input;for(let li=0;li<policy.layers.length;li++){const l=policy.layers[li],out=l.b.map((bias,i)=>{let z=bias,w=l.w[i];for(let j=0;j<v.length;j++)z+=w[j]*v[j];return li===policy.layers.length-1?z:Math.max(0,z)});v=out;}return v;}
async function runGame(){
 if(!policy)await refreshPolicy();
 const job=await (await fetch('/api/job')).json();
 let rows=Array.from({length:4},()=>Array.from({length:4},()=>[])),pen=[0,0,0,0],locks=new Set(),active=0,phase=0,marks=0,turn=0,actions=[];
 const rolls=job.rolls;let dice=rolls[0];
 function actor(){return phase===0||phase===4?active:(active+phase)%4}
 function legal(){let who=actor(),candidate=[];
  if(phase===4){for(let c=0;c<4;c++)for(let w=0;w<2;w++)candidate.push([c,dice[w]+dice[c+2]])}
  else for(let c=0;c<4;c++)candidate.push([c,dice[0]+dice[1]]);
  let out=[];
  for(const [c,n] of candidate){if(locks.has(c))continue;let idx=order[c].indexOf(n),row=rows[who][c];if(idx<0||row.length&&idx<=order[c].indexOf(row[row.length-1]))continue;
   if(idx===10){if(row.length>=5)out.push([c,n,true]);}else out.push([c,n,false]);}
  return out;
 }
 function encode(a){if(a===null)return 0;return 1+22*a[0]+2*order[a[0]].indexOf(a[1])+(a[2]?1:0)}
 function observe(){let v=[];for(let b=0;b<4;b++){for(let c=0;c<4;c++)for(let n of order[c])v.push(rows[b][c].includes(n)?1:0);v.push(pen[b]/4)}
  for(let c=0;c<4;c++)v.push(locks.has(c)?1:0);for(let x of dice)v.push(x/6);
  v.push(active/4,actor()/4,phase/5,Math.min(turn/100,1),marks/2);return v;}
 for(let t=0;t<1000;t++){
  if(locks.size>=2||pen.some(x=>x>=4))break;
  const options=[null,...legal()];let a;
  if(Math.random()<.18)a=options[Math.floor(Math.random()*options.length)];
  else {const q=inference(observe());a=options.reduce((best,x)=>q[encode(x)]>q[encode(best)]?x:best,null);}
  actions.push(encode(a));let who=actor();if(a){rows[who][a[0]].push(a[1]);if(a[2])locks.add(a[0]);if(who===active)marks++}
  if(phase===4&&marks===0)pen[active]++;
  if(locks.size>=2||pen.some(x=>x>=4))break;
  if(phase<4)phase++;else{turn++;active=(active+1)%4;phase=0;marks=0;dice=rolls[turn];if(!dice)throw Error('Dice stream exhausted')}
 }
 const r=await fetch('/api/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticket:job.ticket,moves:actions})});
 if(!r.ok)throw Error((await r.json()).error||'Upload failed');completed++;
}
async function loop(){if(!enabled||busy)return;busy=true;try{await runGame();if(inWorker)postMessage({type:'progress',completed,version:policy.version});else document.getElementById('workerStatus').textContent=`Contributed ${completed} verified games · policy ${policy.version}`;}catch(e){if(inWorker)postMessage({type:'error',message:e.message});else document.getElementById('workerStatus').textContent='Worker error: '+e.message;}finally{busy=false;if(enabled)setTimeout(loop,1000)}}
function toggleWorker(){enabled=!enabled;if(!inWorker)document.getElementById('workerToggle').textContent=enabled?'Stop browser training':'Start browser training';if(enabled)loop()}
if(inWorker)onmessage=e=>{if(e.data==='start'&&!enabled)toggleWorker();if(e.data==='stop'&&enabled)toggleWorker()};
async function stats(){try{const x=await(await fetch('/api/stats')).json();document.getElementById('trainingStats').textContent=`Central games: ${x.central_games} | Browser games: ${x.browser_games} | Updates: ${x.updates} | Replay: ${x.replay_size} | Best win share: ${(100*x.best_win_share).toFixed(1)}% | Sessions: ${x.sessions}`;if(policy&&x.model_version!==policy.version)await refreshPolicy();}catch(e){}}
if(!inWorker){setInterval(stats,5000);stats();}
'''

# Server supplies exact dice sequence; do not trust browser RNG or replay claims.
def rolls_for(seed):
    g=Game(seed=seed)
    return [g.dice]+[tuple(g.rng.randint(1,6) for _ in range(6)) for _ in range(999)]

class Handler(BaseHTTPRequestHandler):
    coord=None
    session=None
    def reply(self,data,status=200,content_type='application/json'):
        body=json.dumps(data).encode() if content_type=='application/json' else data.encode()
        self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path=='/health':return self.reply({'healthy':True})
        if self.path=='/api/history':return self.reply(self.coord.history())
        if self.path=='/api/stats':
            with self.coord.lock:return self.reply(self.coord.stats())
        if self.path=='/api/policy':return self.reply(self.coord.policy_export())
        if self.path=='/api/job':
            if WORKER_TOKEN and self.headers.get('X-Worker-Token')!=WORKER_TOKEN:return self.reply({'error':'Unauthorized'},403)
            try:
                job=self.coord.ticket();job['rolls']=rolls_for(job['seed']);return self.reply(job)
            except ValueError as e:return self.reply({'error':str(e)},429)
        if self.path=='/worker.js':return self.reply(WORKER_JS,content_type='application/javascript')
        if self.path=='/':
            html=GAME_HTML.replace('</body>','<section><h2>Community training</h2><p>Training runs on the server continuously. You can also contribute simulations from this browser while the tab is open.</p><button id="workerToggle" onclick="toggleWorker()">Start browser training</button><p id="workerStatus">Browser training is off</p><p id="trainingStats"></p><section id="training-dashboard"><h2>Training dashboard</h2><p id="trainingUptime"></p><div id="trainingChart" aria-label="Recent game score chart"></div><h3>Recent games</h3><div id="trainingHistory"></div></section></section><script>
let workers=[],completed=0;
function toggleWorker(){
 const button=document.getElementById('workerToggle');
 if(workers.length){workers.forEach(w=>w.terminate());workers=[];button.textContent='Start browser training';document.getElementById('workerStatus').textContent='Browser training stopped';return}
 const count=Math.max(1,Math.min(2,navigator.hardwareConcurrency||1));
 for(let i=0;i<count;i++){const w=new Worker('/worker.js');w.onmessage=e=>{if(e.data.type==='progress'){completed++;document.getElementById('workerStatus').textContent='Contributed '+completed+' verified games'}else if(e.data.type==='error')document.getElementById('workerStatus').textContent='Worker error: '+e.data.message};w.postMessage('start');workers.push(w)}
 button.textContent='Stop browser training';
}
async function stats(){try{const x=await(await fetch('/api/stats')).json();document.getElementById('trainingStats').textContent='Central games: '+x.central_games+' | Browser games: '+x.browser_games+' | Updates: '+x.updates+' | Replay: '+x.replay_size+' | Best win share: '+(100*x.best_win_share).toFixed(1)+'%';document.getElementById('trainingUptime').textContent='Uptime: '+Math.round(x.uptime_seconds/60)+' min | Saved games: '+x.database_games;const games=await(await fetch('/api/history')).json();document.getElementById('trainingHistory').textContent=games.slice(0,10).map(g=>g.source+' '+g.scores.join(' / ')).join(' · ');const scores=games.slice(0,40).reverse().map(g=>Math.max(...g.scores));document.getElementById('trainingChart').innerHTML='<svg role="img" aria-label="Recent winning scores" viewBox="0 0 400 90" style="width:100%;height:110px"><polyline fill="none" stroke="#60a5fa" stroke-width="2" points="'+scores.map((v,i)=>(i*400/Math.max(1,scores.length-1))+','+(85-Math.max(0,Math.min(80,v/2)))).join(' ')+'"/></svg>';}catch(e){}}
setInterval(stats,5000);stats();
</script></body>')
            return self.reply(html,content_type='text/html; charset=utf-8')
        return self.reply({'error':'Not found'},404)
    def do_POST(self):
        if self.path in ('/reset','/move'):
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<0 or n>10000:raise ValueError('Payload too large')
                data=json.loads(self.rfile.read(n) or b'{}')
                if self.path=='/reset':result=self.coord.new_session()
                else:
                    sid=data.get('session_id','')
                    if not isinstance(sid,str):raise ValueError('Invalid session')
                    a=data.get('action')
                    result=self.coord.session_action(sid,None if a is None else tuple(a))
                return self.reply(result)
            except (ValueError,RuntimeError,TypeError) as e:return self.reply({'error':str(e)},400)
        if self.path!='/api/submit':return self.reply({'error':'Not found'},404)
        if WORKER_TOKEN and self.headers.get('X-Worker-Token')!=WORKER_TOKEN:return self.reply({'error':'Unauthorized'},403)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=MAX_BODY:raise ValueError('Invalid body size')
            data=json.loads(self.rfile.read(n));return self.reply(self.coord.accept(data))
        except (ValueError,TypeError,KeyError) as e:return self.reply({'error':str(e)},400)
    def log_message(self,fmt,*args):pass

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8765)
    args=p.parse_args();torch.set_num_threads(max(1,int(os.getenv('QWIXX_TORCH_THREADS','2'))))
    c=Coordinator();Handler.coord=c
    threading.Thread(target=c.run,daemon=True,name='central-trainer').start()
    print(f'Qwixx distributed trainer on {args.host}:{args.port}',flush=True)
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    try:server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:pass
    finally:
        c.stop.set();server.server_close()
        c.checkpoint();c.db.close()
if __name__=='__main__':main()
