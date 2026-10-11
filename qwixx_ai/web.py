"""Local play-versus-AI web UI, standard-library HTTP server."""
import argparse,json,threading
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from .core import Game,COLORS,ORDER
from .agents import heuristic,random_agent
from .model import choose,load

HTML='''<!doctype html><html><head><meta charset="utf-8"><title>Qwixx AI</title><style>

body{font:16px system-ui;background:#111827;color:#f9fafb;max-width:1280px;margin:24px auto;padding:0 16px}
h1{margin-bottom:4px}.hint{color:#9ca3af}button{cursor:pointer;background:#374151;color:white;border:1px solid #6b7280;border-radius:7px;padding:9px 14px;margin:4px}button:hover{background:#4b5563}button.legal{background:#2563eb}a{color:#93c5fd}
details{background:#1f2937;border-radius:8px;padding:14px;margin:24px 0}summary{cursor:pointer}
.dice-tray{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0 22px}.die-wrap{display:flex;flex-direction:column;align-items:center;gap:5px}.die-label{font-size:12px;color:#cbd5e1;font-weight:700}.die{height:66px;width:66px;border-radius:13px;display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,1fr);padding:9px;box-sizing:border-box;box-shadow:inset -5px -6px 8px #0003,0 4px 8px #0005;border:2px solid #ffffff55}.pip{width:11px;height:11px;border-radius:50%;background:#fff;align-self:center;justify-self:center;box-shadow:0 1px 2px #0005}.die.white{background:#f9fafb}.die.white .pip{background:#111827}.die.red{background:#e53e48}.die.yellow{background:#f2bb20}.die.green{background:#35a360}.die.blue{background:#347cda}.die.off{opacity:.3;filter:grayscale(1)}
.scorecard{background:#d1d5db;padding:8px;border-radius:10px;color:#111827;margin:10px 0 18px;overflow-x:auto}.score-row{display:grid;grid-template-columns:repeat(11,minmax(32px,1fr)) 37px;gap:2px;padding:4px;margin-bottom:3px;border-radius:6px;min-width:470px}.score-row.red{background:#d83a46;--ink:#c92f3d;--light:#ffe3e7}.score-row.yellow{background:#f4d934;--ink:#ba8400;--light:#fff9df}.score-row.green{background:#329753;--ink:#267b42;--light:#e4f6e9}.score-row.blue{background:#38568e;--ink:#304e91;--light:#e6eafa}
.number-cell,.lock-cell{margin:0;padding:0;width:100%;box-sizing:border-box;min-height:38px;border:2px solid var(--ink);border-radius:11px;background:var(--light);color:var(--ink);font-size:18px;font-weight:800;display:flex;align-items:center;justify-content:center;position:relative}.number-cell.playable{cursor:pointer;outline:3px solid #fff;outline-offset:-5px;box-shadow:0 0 0 3px #11182766,inset 0 0 0 3px #fff8;animation:available-pulse 1.8s ease-in-out infinite}.number-cell.selected{background:#fff;color:var(--ink);outline:4px solid #111827;box-shadow:0 0 0 4px #f8fafc;animation:none}.phase-controls{background:#1f2937;border-radius:10px;padding:10px 14px;max-width:650px}.phase-controls p{margin:6px 0}.phase-controls button.confirm{background:#167c54;font-weight:800}.phase-controls button:disabled{opacity:.4;cursor:default}.number-cell.playable:hover,.number-cell.playable:focus-visible{background:#fff;color:var(--ink);transform:translateY(-2px);outline:3px solid #111827}.number-cell.playable:disabled{cursor:wait}@keyframes available-pulse{50%{box-shadow:0 0 0 4px #ffffffcc,inset 0 0 0 3px #fff}}.number-cell.marked{background:var(--ink);color:#fff}.number-cell.skipped{opacity:.45}.number-cell.marked:after{content:'✕';font-size:25px;position:absolute;color:#fff}.number-cell.marked .number-label{opacity:.25}.lock-cell{border-radius:50%;font-size:26px}.lock-cell.locked{background:var(--ink);color:white}.score-footer{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:9px 4px}.score-key{display:flex;flex-wrap:wrap;gap:3px}.score-key span{background:#fff;border:1px solid #9ca3af;border-radius:5px;padding:4px 7px;font-size:12px;text-align:center}.penalty-box{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:white;border:2px solid #9ca3af;border-radius:7px;font-weight:900}.penalty-box.used{color:#b91c1c}.score-totals{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-weight:800}.total-box{border:3px solid var(--ink);background:#fff;border-radius:10px;min-width:52px;padding:9px;text-align:center;color:var(--ink)}.total-box.red{--ink:#d83a46}.total-box.yellow{--ink:#c69a00}.total-box.green{--ink:#329753}.total-box.blue{--ink:#38568e}.total-box.penalty{--ink:#6b7280}.total-box.final{--ink:#111827;min-width:75px}.player-heading{margin:16px 0 6px}.turn-info{line-height:1.6}
@media(max-width:650px){body{padding:0 9px}.die{width:54px;height:54px;padding:7px}.pip{width:9px;height:9px}.dice-tray{gap:7px}.scorecard{padding:7px}.score-row{min-width:470px}.number-cell,.lock-cell{min-height:38px}}
</style></head><body><h1>Qwixx AI</h1><p><a href="#training-dashboard">Training dashboard</a> · <a href="#rules-faq">Rules & FAQ</a></p><p class="hint">You play seat 1 against three AI opponents. Click a highlighted number on your scorecard, or skip a phase.</p>
<label for="difficulty">AI difficulty</label><select id="difficulty"><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option><option value="champion" selected>Champion</option></select><button onclick="reset()">New game</button><div id="status"></div><div id="boards"></div><h3>Your choices</h3><div id="actions"></div>
<script>
let state, pending=null;
async function api(path,body){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});let j=await r.json();if(!r.ok){alert(j.error||'Error');return null}return j}
async function reset(){const next=await api('/reset',{difficulty:document.getElementById('difficulty').value});if(next){state=next;pending=null;render()}}
let moving=false;async function act(a){if(moving||!state||state.done)return;moving=true;try{const next=await api('/move',{action:a,session_id:state.session_id});if(next){state=next;pending=null;render()}}finally{moving=false}}

function selectMark(action){if(moving||!state||state.done)return;pending=pending&&pending[0]===action[0]&&pending[1]===action[1]?null:action;render()}
function clearSelection(){pending=null;render()}
function confirmPhase(){act(pending)}
const colors=['red','yellow','green','blue'],names=['Red','Yellow','Green','Blue'];
const playerNames=['You','AI-Sir Rolls-a-Lot','AI-Dicey McDiceface','AI-Count von Qwixx'];
const pipPositions={1:[5],2:[1,9],3:[1,5,9],4:[1,3,7,9],5:[1,3,5,7,9],6:[1,3,4,6,7,9]};
function die(value,color,label,disabled){return '<div class="die-wrap"><div class="die '+color+(disabled?' off':'')+'" aria-label="'+label+' die shows '+value+'">'+pipPositions[value].map(p=>'<span class="pip" style="grid-area:'+Math.ceil(p/3)+' / '+((p-1)%3+1)+'"></span>').join('')+'</div><span class="die-label">'+label+(disabled?' (locked)':'')+'</span></div>'}
function card(board,i){let b='<h3 class="player-heading">'+playerNames[i]+' — '+state.scores[i]+' points · '+board.penalties+' penalties</h3><section class="scorecard" aria-label="'+(i===0?'Your':'AI '+i)+' Qwixx scorecard">';
 board.rows.forEach((row,c)=>{let nums=c<2?[2,3,4,5,6,7,8,9,10,11,12]:[12,11,10,9,8,7,6,5,4,3,2],last=row.length?nums.indexOf(row[row.length-1]):-1,locked=state.locks.includes(colors[c])&&row.includes(nums[10]);
 b+='<div class="score-row '+colors[c]+'" aria-label="'+names[c]+' row">'+nums.map((n,j)=>{const action=i===0&&!state.done?state.legal.find(a=>a[0]===c&&a[1]===n):null;const selected=action&&pending&&pending[0]===c&&pending[1]===n;const classes='number-cell '+(row.includes(n)?'marked':j<last?'skipped':'')+(action?' playable':'')+(selected?' selected':'');const label=names[c]+' '+n+(row.includes(n)?' marked':action?' available to mark':'');return action?'<button type="button" class="'+classes+'" aria-label="'+label+'" onclick="selectMark(['+action.join(',')+'])"><span class="number-label">'+n+'</span></button>':'<div class="'+classes+'" aria-label="'+label+'"><span class="number-label">'+n+'</span></div>'}).join('')+'<div class="lock-cell '+(locked?'locked':'')+'" aria-label="'+names[c]+' padlock">'+(locked?'🔒':'♙')+'</div></div>'});
 b+='<div class="score-footer"><strong>Marks → points:</strong><div class="score-key">'+Array.from({length:12},(_,j)=>'<span>'+(j+1)+'×<br><b>'+((j+1)*(j+2)/2)+'</b></span>').join('')+'</div></div>';
 b+='<div class="score-footer"><strong>Missed throws (−5 each):</strong>'+Array.from({length:4},(_,j)=>'<span class="penalty-box '+(j<board.penalties?'used':'')+'">'+(j<board.penalties?'✕':'')+'</span>').join('')+'</div>';
 b+='<div class="score-footer score-totals"><strong>Result:</strong>'+board.rows.map((row,c)=>{let k=row.length+(state.locks.includes(colors[c])&&row.includes(c<2?12:2)?1:0);return '<span class="total-box '+colors[c]+'">'+(k*(k+1)/2)+'</span>'}).join('<span>+</span>')+'<span>−</span><span class="total-box penalty">'+(board.penalties*5)+'</span><span>=</span><span class="total-box final">'+state.scores[i]+'</span></div></section>';return b}
function render(){if(!state)return;
 const roller=playerNames[state.turn%state.boards.length]||'AI';const whitePhase=state.phase<state.boards.length;
 const instruction=state.done?'Game over':whitePhase?'You may mark the sum of both white dice ('+(state.dice[0]+state.dice[1])+') on any eligible color.':'As the roller, you may mark either white die plus a matching colored die.';
 document.getElementById('status').innerHTML='<p class="turn-info"><strong>Turn '+(state.turn+1)+' — '+roller+' is rolling.</strong> '+(whitePhase?'Everyone can use the white dice.':'Roller may use white + colored dice.')+'</p><div class="dice-tray">'+state.dice.map((n,i)=>die(n,i<2?'white':colors[i-2],i<2?'White '+(i+1):names[i-2],i>=2&&state.locks.includes(colors[i-2]))).join('')+'</div><div class="phase-controls"><strong>'+(whitePhase?'White dice selection':'White + colored selection')+'</strong><p>'+instruction+'</p><p>'+(pending?'Selected: '+names[pending[0]]+' '+pending[1]:'No number selected (skip this phase)')+'</p><button class="confirm" onclick="confirmPhase()" '+(moving?'disabled':'')+'>DONE'+(pending?' — mark selection':' — skip phase')+'</button><button onclick="clearSelection()" '+(!pending?'disabled':'')+'>RESET selection</button></div><p class="turn-info">Locked rows: '+(state.locks.join(', ')||'none')+' · Scores: '+state.scores.join(' / ')+'</p>';

 document.getElementById('boards').innerHTML=state.boards.map(card).join('');
 const choices=state.legal.map(a=>{const c=a[0],n=a[1],details=state.explanations.find(x=>x.move===colors[c]+' '+n),sources=whitePhase?'white '+state.dice[0]+' + white '+state.dice[1]:[0,1].filter(w=>state.dice[w]+state.dice[c+2]===n).map(w=>'white '+(w+1)+' ('+state.dice[w]+') + '+colors[c]+' ('+state.dice[c+2]+')').join(' OR ');return '<li><strong>'+names[c]+' '+n+(a[2]?' 🔒':'')+'</strong> — '+sources+(details?' · +'+details.immediate+' immediate points · skips '+details.skipped:'')+'</li>'}).join('');
 document.getElementById('actions').innerHTML=state.done?'<b>Game over. Start a new game.</b>':'<p>Select a highlighted number on your scorecard, then press DONE under the dice. Press DONE without a selection to skip this phase.</p><details class="move-analysis"><summary>Move analysis — '+state.legal.length+' available scorecard moves</summary><ul>'+choices+'</ul><p>Skip phase: press DONE without a selection. Equivalent dice combinations leading to the same number are shown together.</p></details>';

}

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
                    if self.path=='/reset':
                        difficulty=data.get('difficulty','champion')
                        if difficulty not in ('beginner','intermediate','advanced','champion'):raise ValueError('Invalid difficulty')
                        session.difficulty=difficulty;session.game=Game();session.advance()
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
