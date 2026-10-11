"""Local play-versus-AI web UI, standard-library HTTP server."""
import argparse,json,threading
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from .core import Game,COLORS,ORDER
from .agents import heuristic,random_agent
from .model import choose,load

HTML='''<!doctype html><html><head><meta charset="utf-8"><title>Qwixx AI</title><style>
body{font:16px system-ui;background:#111827;color:#f9fafb;max-width:920px;margin:30px auto;padding:0 16px}
h1{margin-bottom:4px}.hint{color:#9ca3af}.row{display:flex;align-items:center;gap:7px;margin:12px 0}
.cell{padding:10px;border-radius:8px;background:#374151;min-width:32px;text-align:center}
.mark{background:#0f766e}.red{color:#f87171}.yellow{color:#facc15}.green{color:#4ade80}.blue{color:#60a5fa}
button{cursor:pointer;background:#374151;color:white;border:1px solid #6b7280;border-radius:7px;padding:9px 14px;margin:4px}
details{background:#1f2937;border-radius:8px;padding:14px;margin:24px 0}summary{cursor:pointer}a{color:#93c5fd}button:hover{background:#4b5563}.legal{background:#2563eb} .dice{font-size:24px;letter-spacing:6px}
</style></head><body><h1>Qwixx AI</h1><p><a href="#training-dashboard">Training dashboard</a> · <a href="#rules-faq">Rules & FAQ</a></p><p class="hint">You play seat 1 against three AI opponents. Click a legal move or Pass.</p>
<label for="difficulty">AI difficulty</label><select id="difficulty"><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option><option value="champion" selected>Champion</option></select><button onclick="reset()">New game</button><div id="status"></div><div id="boards"></div><h3>Your choices</h3><div id="actions"></div>
<script>
let state;
async function api(path,body){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});let j=await r.json();if(!r.ok)alert(j.error||'Error');return j}
async function reset(){state=await api('/reset',{difficulty:document.getElementById('difficulty').value});render()}
async function act(a){state=await api('/move',{action:a,session_id:state.session_id});render()}
function render(){if(!state)return;
 document.getElementById('status').innerHTML=`<p class="dice">🎲 ${state.dice.join(' · ')}</p><p>Turn ${state.turn+1} · ${state.done?'Game over':state.actor===0?'Your move':'AI thinking'} · Phase ${state.phase} · Locked: ${state.locks.join(', ')||'none'} · Colored dice in play: ${['red','yellow','green','blue'].filter(c=>!state.locks.includes(c)).join(', ')}</p><p>Scores: ${state.scores.join(' / ')}</p>`;
 let b='';state.boards.forEach((board,i)=>{b+=`<h3>${i===0?'You':'AI '+i} — ${state.scores[i]} points · ${board.penalties} penalties</h3>`;
 board.rows.forEach((row,c)=>{let nums=c<2?[2,3,4,5,6,7,8,9,10,11,12]:[12,11,10,9,8,7,6,5,4,3,2];b+=`<div class="row ${['red','yellow','green','blue'][c]}"><b>${['Red','Yellow','Green','Blue'][c]}</b>`+nums.map(n=>`<span class="cell ${row.includes(n)?'mark':''}">${row.includes(n)?'✓':n}</span>`).join('')+'</div>'})});document.getElementById('boards').innerHTML=b;
 document.getElementById('actions').innerHTML=(state.explanations?'<h4>Move analysis</h4><p>'+state.explanations.map(x=>x.move+': +'+x.immediate+' points, skips '+x.skipped+' numbers'+(x.lock?', locks row':'')).join(' · ')+'</p>':'')+(state.done?'<b>Final scores above. Start a new game.</b>':`<button onclick="act(null)">Pass</button>`+state.legal.map(a=>`<button class="legal" onclick='act(${JSON.stringify(a)})'>${['Red','Yellow','Green','Blue'][a[0]]} ${a[1]} ${a[2]?'🔒':''}</button>`).join(''));}
reset();</script><details id="rules-faq"><summary>Qwixx rules & FAQ</summary>
<h3>White dice (everyone)</h3><p>Every player may mark one number matching the sum of the two white dice, or pass. These decisions are simultaneous, including row locks.</p>
<h3>Colored die (active player only)</h3><p>The active player may combine either white die with one available colored die and mark that row, or pass.</p>
<h3>What does left to right mean?</h3><p>Numbers must be marked strictly from left to right. Skipped numbers are permanently unavailable.</p>
<h3>What is a misthrow penalty?</h3><p>Only the active player takes a misthrow penalty when neither phase produces a mark. Each penalty costs five points.</p>
<h3>How does the padlock work?</h3><p>After five prior marks in a row, marking its far-right number also marks the padlock, closes that color to everyone and removes the colored die. Multiple players can lock during the same white-dice phase.</p>
<h3>When does the game end?</h3><p>After the second color is locked or any player receives their fourth misthrow penalty. Resolve simultaneous white-dice choices first.</p>
<h3>How is scoring calculated?</h3><p>Each row scores triangular points based on its marks, including the padlock bonus mark. Subtract five points per penalty. The highest score wins; ties share the win.</p>
</details></body></html>'''

class Session:
    def __init__(self,model=None,difficulty='champion'):
        if difficulty not in ('beginner','intermediate','advanced','champion'):raise ValueError('Invalid difficulty')
        self.model=model;self.difficulty=difficulty;self.lock=threading.Lock();self.game=Game()
    def ai(self,g):
        if self.difficulty=='beginner':return random_agent(g)
        if self.difficulty=='intermediate':return heuristic(g)
        if self.model is None:return heuristic(g)
        return choose(self.model,g,epsilon=0.18 if self.difficulty=='advanced' else 0)
    def advance(self):
        while not self.game.done and self.game.actor!=0:self.game.step(self.ai(self.game))
    def explanations(self):
        g=self.game;b=g.boards[g.actor]
        out=[]
        for c,n,locking in g.legal():
            row=b.rows[c];index=ORDER[c].index(n)
            skipped=index-(ORDER[c].index(row[-1]) if row else -1)-1
            k=len(row)
            immediate=(k+1)*(k+2)//2-k*(k+1)//2+(k+2 if locking else 0)
            out.append({'move':f'{COLORS[c]} {n}','immediate':immediate,'skipped':skipped,'lock':locking})
        return out
    def view(self):
        g=self.game
        return {'turn':g.turn,'phase':g.phase,'actor':g.actor,'dice':g.dice,'locks':[COLORS[c] for c in sorted(g.locks)],'done':g.done,'scores':g.scores(),'boards':[{'rows':b.rows,'penalties':b.penalties} for b in g.boards],'legal':g.legal() if not g.done else [],'difficulty':self.difficulty,'explanations':self.explanations() if not g.done else []}

def serve(host='127.0.0.1',port=8765,model_path=None):
    session=Session(load(model_path) if model_path else None)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/':self.send_error(404);return
            body=HTML.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if self.path not in ('/reset','/move'):self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length>10000:raise ValueError('Payload too large')
                data=json.loads(self.rfile.read(length) or b'{}')
                with session.lock:
                    if self.path=='/reset':session.game=Game();session.advance()
                    else:
                        a=data.get('action');a=None if a is None else tuple(a)
                        session.game.step(a);session.advance()
                    result=session.view()
                body=json.dumps(result).encode();status=200
            except (ValueError,RuntimeError,KeyError,TypeError) as e:
                body=json.dumps({'error':str(e)}).encode();status=400
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    print(f'Open http://{host}:{port}');ThreadingHTTPServer((host,port),Handler).serve_forever()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);p.add_argument('--model');a=p.parse_args();serve(a.host,a.port,a.model)
