"""Local play-versus-AI web UI, standard-library HTTP server."""
import argparse,json,threading,os,html,copy
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from .core import Game,COLORS,ORDER
from .agents import heuristic,random_agent
from .model import choose,load
from .champion import champion
from .heuristic_v2 import heuristic_v2

# Supplied by the image build; no Git repository is required at runtime.
GIT_VERSION = os.getenv('QWIXX_GIT_VERSION', '').strip()

HTML='''<!doctype html><html><head><meta charset="utf-8"><title>Qwixx AI</title><style>

body{font:16px system-ui;background:#111827;color:#f9fafb;max-width:1280px;margin:24px auto;padding:0 16px}
h1{margin-bottom:4px}.hint{color:#9ca3af}button{cursor:pointer;background:#374151;color:white;border:1px solid #6b7280;border-radius:7px;padding:9px 14px;margin:4px}button:hover{background:#4b5563}button.legal{background:#2563eb}a{color:#93c5fd}
details{background:#1f2937;border-radius:8px;padding:14px;margin:24px 0}summary{cursor:pointer}
.dice-tray{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0 22px}.die-wrap{display:flex;flex-direction:column;align-items:center;gap:5px}.die-label{font-size:12px;color:#cbd5e1;font-weight:700}.die{height:66px;width:66px;border-radius:13px;display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,1fr);padding:9px;box-sizing:border-box;box-shadow:inset -5px -6px 8px #0003,0 4px 8px #0005;border:2px solid #ffffff55}.pip{width:11px;height:11px;border-radius:50%;background:#fff;align-self:center;justify-self:center;box-shadow:0 1px 2px #0005}.die.white{background:#f9fafb}.die.white .pip{background:#111827}.die.red{background:#e53e48}.die.yellow{background:#f2bb20}.die.green{background:#35a360}.die.blue{background:#347cda}.die.off{opacity:.65;filter:saturate(.55) brightness(.88)}.die.off .pip{opacity:.7}.die{position:relative}.die.locked-die:after{content:'🔒';position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:34px;background:#11182788;border-radius:11px}.game-over-overlay{position:fixed;inset:0;z-index:1000;background:#000a;display:flex;align-items:center;justify-content:center;padding:16px}.game-over-dialog{background:#1f2937;border:2px solid #94a3b8;border-radius:16px;box-shadow:0 20px 70px #0009;padding:24px;max-width:420px;width:100%;text-align:center}.game-over-dialog h2{margin-top:0}.game-over-dialog button{background:#167c54;font-weight:800}
.scorecard{width:min(100%,640px);box-sizing:border-box;background:#d1d5db;padding:8px;border-radius:10px;color:#111827;margin:10px 0 18px;overflow-x:auto}.score-row{display:grid;grid-template-columns:repeat(11,minmax(24px,1fr)) 30px;gap:2px;padding:4px;margin-bottom:3px;border-radius:6px;min-width:360px}.score-row.red{background:#d83a46;--ink:#c92f3d;--light:#ffe3e7}.score-row.yellow{background:#f4d934;--ink:#ba8400;--light:#fff9df}.score-row.green{background:#329753;--ink:#267b42;--light:#e4f6e9}.score-row.blue{background:#38568e;--ink:#304e91;--light:#e6eafa}
.number-cell,.lock-cell{margin:0;padding:0;width:100%;box-sizing:border-box;min-height:38px;border:2px solid var(--ink);border-radius:11px;background:var(--light);color:var(--ink);font-size:15px;font-weight:800;display:flex;align-items:center;justify-content:center;position:relative}.number-cell.playable{cursor:pointer;outline:3px solid #fff;outline-offset:-5px;box-shadow:0 0 0 3px #11182766,inset 0 0 0 3px #fff8;animation:available-pulse 1.8s ease-in-out infinite}.number-cell.selected{background:#fff;color:var(--ink);outline:4px solid #111827;box-shadow:0 0 0 4px #f8fafc;animation:none}.phase-controls{background:#1f2937;border-radius:10px;padding:10px 14px;max-width:650px}.phase-controls p{margin:6px 0}.phase-controls button.confirm{background:#167c54;font-weight:800}.strike-warning{color:#fbbf24;font-weight:700}.phase-controls button:disabled{opacity:.4;cursor:default}.number-cell.playable:hover,.number-cell.playable:focus-visible{background:#fff;color:var(--ink);transform:translateY(-2px);outline:3px solid #111827}.number-cell.playable:disabled{cursor:wait}@keyframes available-pulse{50%{box-shadow:0 0 0 4px #ffffffcc,inset 0 0 0 3px #fff}}.number-cell.marked{background:var(--ink);color:#fff}.number-cell.skipped{opacity:.45}.number-cell.marked:after{content:'✕';font-size:25px;position:absolute;color:#fff}.number-cell.marked .number-label{opacity:.25}.lock-cell{border-radius:50%;font-size:26px}.lock-cell.locked{background:var(--ink);color:white}.score-footer{display:flex;flex-wrap:wrap;align-items:center;gap:5px;padding:5px 2px;font-size:12px}.score-footer.combined{justify-content:space-between}.score-footer .missed-throws{display:flex;align-items:center;gap:4px}.score-key{display:flex;flex-wrap:wrap;gap:3px}.score-key span{background:#fff;border:1px solid #9ca3af;border-radius:5px;padding:3px 4px;font-size:10px;text-align:center}.penalty-box{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;background:white;border:2px solid #9ca3af;border-radius:7px;font-weight:900}.penalty-box.used{color:#b91c1c}.score-totals{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-weight:800}.total-box{border:3px solid var(--ink);background:#fff;border-radius:10px;min-width:52px;padding:9px;text-align:center;color:var(--ink)}.total-box.red{--ink:#d83a46}.total-box.yellow{--ink:#c69a00}.total-box.green{--ink:#329753}.total-box.blue{--ink:#38568e}.total-box.penalty{--ink:#6b7280}.total-box.final{--ink:#111827;min-width:75px}.player-heading{margin:16px 0 6px}.player-panel.inactive{opacity:.42;filter:grayscale(.85)}.player-panel.inactive .scorecard{pointer-events:none}.player-panel.inactive .number-cell.playable{pointer-events:auto}.player-panel{max-width:640px}.turn-info{line-height:1.6}
@media(max-width:650px){body{padding:0 9px}.die{width:54px;height:54px;padding:7px}.pip{width:9px;height:9px}.dice-tray{gap:7px}.scorecard{padding:7px}.score-row{min-width:470px}.number-cell,.lock-cell{min-height:38px}}
 .career-panel{background:#1f2937;border:1px solid #64748b;border-radius:12px;padding:16px;max-width:680px;margin:12px 0}.career-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:8px}.career-stat{background:#111827;padding:10px;border-radius:8px}.career-stat b{display:block;font-size:22px}.career-panel table{width:100%;border-collapse:collapse}.career-panel td,.career-panel th{padding:6px;border-bottom:1px solid #374151;text-align:left}.career-panel input,.career-panel select{padding:8px;border-radius:6px;max-width:180px} .turn-banner{max-width:650px;border:3px solid #4ade80;background:#123a2b;color:#f0fdf4;border-radius:14px;padding:16px;margin:14px 0}.turn-banner h2{font-size:27px;margin:0 0 6px}.turn-banner p{margin:0}.turn-banner.second{border-color:#fbbf24;background:#42310d}.white-total{font-size:28px;font-weight:900;background:#fff;color:#111827;border-radius:10px;display:inline-block;padding:8px 14px;margin-top:10px}.phase-controls{border:2px solid #4ade80}.phase-controls.second{border-color:#fbbf24}.player-panel.human-active .scorecard{box-shadow:0 0 0 4px #4ade80}.player-panel.human-active .player-heading{color:#86efac}.number-cell.playable{box-shadow:0 0 0 4px #22c55e,inset 0 0 0 3px #fff}.number-cell.selected{box-shadow:0 0 0 4px #fbbf24;animation:none}@media(prefers-reduced-motion:reduce){.number-cell.playable{animation:none}}.score-row.row-locked{position:relative;background:#17212e!important;border:3px solid #e2e8f0;box-shadow:inset 0 0 0 2px #0f172a;gap:2px;padding-top:28px}.score-row.row-locked:before{content:'🔒 ROW LOCKED — NO MORE MARKS';position:absolute;top:3px;left:8px;color:#f8fafc;font-size:12px;font-weight:900;letter-spacing:.06em}.score-row.row-locked .number-cell{opacity:.68;filter:grayscale(.65)}.score-row.row-locked .number-cell.marked{opacity:1;filter:none}.score-row.row-locked .lock-cell{background:#e2e8f0;color:#0f172a;border-color:#e2e8f0}.score-row.row-locked .number-cell.playable{animation:none;box-shadow:none;cursor:default}.score-row.row-locked .row-lock-overlay{position:absolute;inset:28px 4px 3px;z-index:4;pointer-events:none;display:flex;align-items:center;justify-content:center}.score-row.row-locked .row-lock-overlay:before{content:'';position:absolute;left:0;right:0;top:50%;height:5px;transform:translateY(-50%);background:#f8fafc;box-shadow:0 1px 3px #0f172a}.score-row.row-locked .row-lock-overlay span{position:relative;padding:4px 18px;border:2px solid #f8fafc;border-radius:6px;background:#17212e;color:#fff;font-size:16px;font-weight:950;letter-spacing:.15em;box-shadow:0 2px 6px #0f172a}.roller-header{display:flex;flex-wrap:wrap;align-items:center;gap:10px;max-width:680px;margin:14px 0 4px}.turn-count{font-size:15px;font-weight:800;color:#cbd5e1}.roller-name{font-size:clamp(19px,3vw,26px);font-weight:900;padding:6px 13px;border-radius:9px}.roller-name.human{background:#166534;color:#f0fdf4;border:2px solid #4ade80}.roller-name.opponent{background:#334155;color:#fff;border:2px solid #64748b}.roller-caption{font-size:16px;color:#cbd5e1}.turn-subtitle{margin:5px 0 10px;color:#cbd5e1}.phase-controls{border:1px solid #475569!important}.phase-controls.second{border-color:#64748b!important}.player-panel.human-active .scorecard{box-shadow:none}.player-panel.human-active .player-heading{color:inherit}.app-tabs{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0;border-bottom:1px solid #475569;padding-bottom:8px}.app-tabs button{margin:0;background:transparent;border-color:transparent;font-weight:700}.app-tabs button[aria-selected="true"]{background:#334155;border-color:#64748b}.app-panel[hidden]{display:none!important}.app-panel{padding-top:8px}.app-panel .career-panel{max-width:900px}.game-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:8px}.app-panel h2{margin-top:10px}@media(max-width:650px){.app-tabs button{font-size:13px;padding:8px}}</style></head><body><h1>Qwixx AI <small id="app-version" style="font-size:.48em;color:#94a3b8;font-weight:600"></small></h1><p><a href="/training">Training dashboard</a> · <a href="#rules-faq">Rules & FAQ</a></p><p class="hint">You play seat 1 against three AI opponents. Click a highlighted number on your scorecard, or skip a phase.</p>
<label for="difficulty">AI difficulty</label><select id="difficulty"><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option><option value="champion" selected>Champion</option></select><button onclick="reset()">New game</button> <span id="account-status">Guest</span> <button onclick="toggleStats()">Your Stats</button> <button id="auth-logout" onclick="logout()" hidden>Sign out</button><div id="auth-login"><label>Email <input id="auth-email" type="email" placeholder="you@example.com"></label><label> Name (new accounts) <input id="auth-name" placeholder="Display name"></label><button onclick="requestCode()">Email me a code</button><div id="verify-box" hidden><label>Six-digit code <input id="auth-code" inputmode="numeric" maxlength="6"></label><button onclick="verifyCode()">Verify & sign in</button></div><p id="auth-message"></p></div><section id="career" class="career-panel" hidden></section><p id="game-error" role="alert" style="background:#7f1d1d;padding:12px;border-radius:8px" hidden></p><div id="status"><p>Loading game…</p></div><div id="boards"></div><h3>Your choices</h3><div id="actions"></div><details><summary>Turn action history (diagnostics)</summary><pre id="game-history" style="white-space:pre-wrap"></pre></details>
<script>
(function(){
const sections=[['play','Play'],['account','Account & Stats'],['analysis','Move Analysis'],['training','Training'],['rules','Rules & FAQ']];
function activate(id){
 for(const [key] of sections){const panel=document.getElementById('section-'+key),tab=document.getElementById('tab-'+key);if(!panel||!tab)continue;panel.hidden=key!==id;tab.setAttribute('aria-selected',String(key===id));tab.tabIndex=key===id?0:-1}
 if(id==='account'&&typeof verifiedPlayer!=='undefined'&&verifiedPlayer&&typeof showStats==='function')showStats();
 if(id==='training'&&typeof drawDashboard==='function')drawDashboard();
}
window.showTab=activate;
document.addEventListener('DOMContentLoaded',()=>{
 const heading=document.querySelector('body > h1');if(!heading)return;
 const nav=document.createElement('nav');nav.className='app-tabs';nav.setAttribute('role','tablist');nav.setAttribute('aria-label','Qwixx sections');
 const panels={};
 for(const [key,label] of sections){
  const tab=document.createElement('button');tab.type='button';tab.id='tab-'+key;tab.textContent=label;tab.setAttribute('role','tab');tab.setAttribute('aria-controls','section-'+key);tab.addEventListener('click',()=>activate(key));nav.appendChild(tab);
  const panel=document.createElement('section');panel.id='section-'+key;panel.className='app-panel';panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',tab.id);panel.hidden=true;panels[key]=panel;
 }
 heading.after(nav,...sections.map(([key])=>panels[key]));
 const move=(id,key)=>{const el=document.getElementById(id);if(el)panels[key].appendChild(el)};
 const moveSelector=(selector,key)=>{const el=document.querySelector(selector);if(el)panels[key].appendChild(el)};
 const toolbar=document.createElement('div');toolbar.className='game-toolbar';panels.play.appendChild(toolbar);
 for(const selector of ['label[for="difficulty"]','#difficulty','button[onclick="reset()"]']){const el=document.querySelector(selector);if(el)toolbar.appendChild(el)}
 for(const id of ['game-error','status','boards'])move(id,'play');
 for(const id of ['account-status','auth-logout','auth-login','career'])move(id,'account');
 const statsButton=document.querySelector('button[onclick="toggleStats()"]');if(statsButton)statsButton.remove();
 const choices=document.querySelector('#actions');if(choices){const heading=choices.previousElementSibling;if(heading&&heading.tagName==='H3')panels.analysis.appendChild(heading);panels.analysis.appendChild(choices)}
 const history=document.getElementById('game-history');if(history&&history.closest('details'))panels.analysis.appendChild(history.closest('details'));
 const rules=document.getElementById('rules-faq');if(rules)panels.rules.appendChild(rules);
 const training=document.getElementById('training-dashboard');
 if(training){let node=training;while(node){const next=node.nextSibling;if(node.nodeType===1&&node.tagName==='SCRIPT')break;panels.training.appendChild(node);node=next}}
 else{const note=document.createElement('p');note.textContent='Training dashboard requires the distributed server.';panels.training.appendChild(note)}
 const oldLinks=document.querySelector('a[href="/training"]');if(oldLinks&&oldLinks.parentElement&&oldLinks.parentElement.tagName==='P')oldLinks.parentElement.remove();
 const intro=document.querySelector('body > p.hint');if(intro)intro.remove();
 activate('play');
});
})();
</script>
<script>
let state, pending=null, gameOverShown=false;

async function api(path,body){try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});const j=await r.json();if(!r.ok)throw Error(j.error||'Request failed ('+r.status+')');return j}catch(e){if(path.startsWith('/api/auth/')){const msg=document.getElementById('auth-message');if(msg)msg.textContent='Email verification error: '+e.message+'. Check SMTP settings and container logs.';console.error('Qwixx authentication:',e)}else showGameError(e.message);return null}}
function showGameError(message){const el=document.getElementById('game-error');if(el){el.hidden=false;el.textContent='Game error: '+message+' — please try New game or reload the page.'}console.error('Qwixx:',message)}
async function reset(){const next=await api('/reset',{difficulty:document.getElementById('difficulty').value});if(next){document.getElementById('game-error').hidden=true;state=next;pending=null;gameOverShown=false;document.getElementById('game-over-modal')?.remove();render()}}
let verifiedPlayer=null,verificationEmail='';
async function loadPlayers(){try{const response=await fetch('/api/auth/me');const data=await response.json();verifiedPlayer=data.player;document.getElementById('account-status').textContent=verifiedPlayer?'Signed in: '+verifiedPlayer.name:'Guest — games will not be saved';document.getElementById('auth-login').hidden=!!verifiedPlayer;document.getElementById('auth-logout').hidden=!verifiedPlayer}catch(e){console.warn('Account unavailable',e)}}
async function requestCode(){document.getElementById('auth-message').textContent='Requesting verification email…';const email=document.getElementById('auth-email').value.trim();const name=document.getElementById('auth-name').value.trim();const response=await api('/api/auth/request',{email,name:name||null});if(response){verificationEmail=email;document.getElementById('verify-box').hidden=false;document.getElementById('auth-message').textContent='Code sent. Check your email (expires in 10 minutes).'}}
async function verifyCode(){const code=document.getElementById('auth-code').value.trim();const response=await api('/api/auth/verify',{email:verificationEmail,code});if(response){document.getElementById('auth-message').textContent='Email verified. Start a new game to track stats.';document.getElementById('verify-box').hidden=true;await loadPlayers();if(!document.getElementById('career').hidden)showStats()}}
async function logout(){await api('/api/auth/logout',{});verifiedPlayer=null;await loadPlayers();document.getElementById('career').hidden=true}
function toggleStats(){const panel=document.getElementById('career');panel.hidden=!panel.hidden;if(!panel.hidden)showStats()}
async function showStats(){const panel=document.getElementById('career');if(!verifiedPlayer){panel.textContent='Register or sign in with your email to track completed games.';return}try{const response=await fetch('/api/player-stats');if(!response.ok)throw Error('Unable to load player stats');const data=await response.json();const fields=[['Games',data.games],['Wins',data.wins],['Losses',data.losses],['Ties',data.ties],['Win rate',data.win_rate+'%'],['Average score',data.average_score],['Highest score',data.highest_score??'—'],['Lowest score',data.lowest_score??'—'],['Average penalties',data.average_penalties],['Average turns',data.average_turns],['Current win streak',data.current_win_streak],['Longest win streak',data.longest_win_streak]];panel.innerHTML='<h3>Career stats — '+escapeHtml(data.player.name)+'</h3><div class="career-grid">'+fields.map(([label,value])=>'<div class="career-stat">'+escapeHtml(label)+'<b>'+value+'</b></div>').join('')+'</div><h4>By difficulty</h4>'+Object.entries(data.by_difficulty).map(([level,x])=>'<p>'+escapeHtml(level)+': '+x.wins+' wins / '+x.games+' games · average '+x.average_score+'</p>').join('')+'<h4>Recent completed games</h4><table><thead><tr><th>Difficulty</th><th>Score</th><th>Result</th><th>Penalties</th></tr></thead><tbody>'+data.recent.map(g=>'<tr><td>'+escapeHtml(g.difficulty)+'</td><td>'+g.score+'</td><td>'+g.result+'</td><td>'+g.penalties+'</td></tr>').join('')+'</tbody></table>'}catch(e){panel.textContent=e.message}}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
let moving=false;async function act(a){if(moving||!state||state.done)return;if(state.phase<state.boards.length&&state.roller===0&&a===null&&state.legal.length===0&&!window.confirm('Pass on the white dice total '+(state.dice[0]+state.dice[1])+'? You can still use white + colored dice in Step 2. A strike only occurs if you also pass on Step 2.'))return;moving=true;try{const next=await api('/move',{action:a,session_id:state.session_id,expected_turn:state.turn,expected_phase:state.phase});if(next){state=next;pending=null;render()}}finally{moving=false}}

function selectMark(action){if(moving||!state||state.done)return;pending=pending&&pending[0]===action[0]&&pending[1]===action[1]?null:action;render()}
function clearSelection(){pending=null;render()}
function confirmPhase(){if(pending)act(pending)}
async function backToStep1(){if(moving||!state||state.done||!state.can_undo_step1)return;moving=true;try{const next=await api('/undo-step1',{session_id:state.session_id,expected_turn:state.turn,expected_phase:state.phase});if(next){state=next;pending=null;render()}}finally{moving=false}}
function skipPhase(){if(!state||state.done||moving)return;const strike=state.roller===0&&state.phase===state.boards.length&&state.active_marks===0;if(strike&&state.legal.length===0&&!window.confirm('No legal marks available. Skipping gives you a strike (−5 points). Continue?'))return;act(null)}
const colors=['red','yellow','green','blue'],names=['Red','Yellow','Green','Blue'];
const playerNames=['You','AI-Sir Rolls-a-Lot','AI-Dicey McDiceface','AI-Count von Qwixx'];
const pipPositions={1:[5],2:[1,9],3:[1,5,9],4:[1,3,7,9],5:[1,3,5,7,9],6:[1,3,4,6,7,9]};
function die(value,color,label,disabled){const locked=state.locks.includes(color);return '<div class="die-wrap"><div class="die '+color+(disabled?' off':'')+(locked?' locked-die':'')+'" aria-label="'+label+' die shows '+value+(locked?', locked':'')+'">'+pipPositions[value].map(p=>'<span class="pip" style="grid-area:'+Math.ceil(p/3)+' / '+((p-1)%3+1)+'"></span>').join('')+'</div><span class="die-label">'+label+'</span></div>'}
function card(board,i){let b='<div class="player-panel '+(i!==0&&i!==state.roller?'inactive':'active')+'"><h3 class="player-heading">'+playerNames[i]+' — '+state.scores[i]+' points · '+board.penalties+' penalties</h3><section class="scorecard" aria-label="'+(i===0?'Your':'AI '+i)+' Qwixx scorecard">';
 board.rows.forEach((row,c)=>{let nums=c<2?[2,3,4,5,6,7,8,9,10,11,12]:[12,11,10,9,8,7,6,5,4,3,2],last=row.length?nums.indexOf(row[row.length-1]):-1,locked=state.locks.includes(colors[c]),personalLock=locked&&row.includes(nums[10]);
 b+='<div class="score-row '+colors[c]+(locked?' row-locked':'')+'" aria-label="'+names[c]+' row'+(locked?' locked':'')+'">'+nums.map((n,j)=>{const action=i===0&&!state.done&&!locked?state.legal.find(a=>a[0]===c&&a[1]===n):null;const selected=action&&pending&&pending[0]===c&&pending[1]===n;const classes='number-cell '+(row.includes(n)?'marked':j<last?'skipped':'')+(action?' playable':'')+(selected?' selected':'');const label=names[c]+' '+n+(row.includes(n)?' marked':action?' available to mark':'');return action?'<button type="button" class="'+classes+'" aria-label="'+label+'" onclick="selectMark(['+action.join(',')+'])"><span class="number-label">'+n+'</span></button>':'<div class="'+classes+'" aria-label="'+label+'"><span class="number-label">'+n+'</span></div>'}).join('')+'<div class="lock-cell '+(locked?'locked':'')+'" aria-label="'+names[c]+' padlock">'+(locked?'🔒':'♙')+'</div>'+(locked?'<div class="row-lock-overlay" aria-hidden="true"><span>LOCKED</span></div>':'')+'</div>'});
 b+='<div class="score-footer combined"><strong>Marks → points:</strong><div class="score-key">'+Array.from({length:12},(_,j)=>'<span>'+(j+1)+'×<br><b>'+((j+1)*(j+2)/2)+'</b></span>').join('')+'</div><div class="missed-throws"><strong>Missed throws (−5 each):</strong>'+Array.from({length:4},(_,j)=>'<span class="penalty-box '+(j<board.penalties?'used':'')+'">'+(j<board.penalties?'✕':'')+'</span>').join('')+'</div></div>';
 b+='<div class="score-footer score-totals"><strong>Result:</strong>'+board.rows.map((row,c)=>{let k=row.length+(state.locks.includes(colors[c])&&row.includes(c<2?12:2)?1:0);return '<span class="total-box '+colors[c]+'">'+(k*(k+1)/2)+'</span>'}).join('<span>+</span>')+'<span>−</span><span class="total-box penalty">'+(board.penalties*5)+'</span><span>=</span><span class="total-box final">'+state.scores[i]+'</span></div></section></div>';return b}
function closeGameOver(){document.getElementById('game-over-modal')?.remove()}
function showGameOver(){if(gameOverShown||!state.done)return;gameOverShown=true;const rankings=state.scores.map((score,i)=>({name:playerNames[i],score})).sort((a,b)=>b.score-a.score);const overlay=document.createElement('div');overlay.id='game-over-modal';overlay.className='game-over-overlay';overlay.setAttribute('role','dialog');overlay.setAttribute('aria-modal','true');overlay.setAttribute('aria-label','Game over');overlay.innerHTML='<div class="game-over-dialog"><h2>🏁 Game over!</h2><p><strong>'+rankings[0].name+' wins!</strong></p><p>'+rankings.map(p=>p.name+': '+p.score).join('<br>')+'</p><button onclick="reset()">Play again</button><button onclick="closeGameOver()">View final scorecards</button></div>';document.body.appendChild(overlay)}
function render(){if(!state)return;
 const roller=playerNames[state.roller]||'AI';const whitePhase=state.phase<state.boards.length;const youRoll=state.roller===0;
 const instruction=state.done?'Game over':whitePhase?'White dice: '+state.dice[0]+' + '+state.dice[1]+' = '+(state.dice[0]+state.dice[1]):'Choose one white die plus a matching colored die.';
 const rollerLabel=youRoll?'YOU':roller;
 const turnHeader='<div class="roller-header"><span class="turn-count">TURN '+(state.turn+1)+'</span><span class="roller-name '+(youRoll?'human':'opponent')+'">'+rollerLabel+'</span><span class="roller-caption">'+(youRoll?'are rolling':'is rolling')+'</span></div>';
  document.getElementById('status').innerHTML=turnHeader+'<p class="turn-subtitle">'+(whitePhase?'Step 1 · Everyone may mark the white-dice total':'Step 2 · Roller may use white + colored dice')+'</p><div class="dice-tray">'+state.dice.map((n,i)=>die(n,i<2?'white':colors[i-2],i<2?'White '+(i+1):names[i-2],i>=2&&(!youRoll||whitePhase||state.locks.includes(colors[i-2])))).join('')+'</div><div class="phase-controls"><p class="phase-instruction">'+instruction+'</p>'+(pending?'<p class="selected-mark">Selected: '+names[pending[0]]+' '+pending[1]+'</p>':'')<button class="confirm" onclick="confirmPhase()" '+(!pending||moving?'disabled':'')+'>✓ Mark selected number</button><button onclick="clearSelection()" '+(!pending?'disabled':'')+'>Clear selection</button><button class="skip" onclick="skipPhase()">Skip opportunity →</button>'+(state.can_undo_step1?'<button onclick="backToStep1()" '+(moving?'disabled':'')+'>← Back to Step 1</button>':'')+(youRoll&&!whitePhase&&state.active_marks===0?'<p class="strike-warning">⚠ Skipping now gives you a strike (−5 points).</p>':youRoll&&whitePhase?'<p class="hint">A strike only occurs if you skip both opportunities on your roll.</p>' :'<p class="hint">No strike when you pass on another player’s roll.</p>')+'</div><p class="turn-info">Locked rows: '+(state.locks.join(', ')||'none')+' · Scores: '+state.scores.join(' / ')+'</p>';

 document.getElementById('boards').innerHTML=state.boards.map(card).join('');
 const history=document.getElementById('game-history');
 if(history){const actions=state.action_history||[];history.textContent=actions.length?actions.slice(-12).map(e=>'Turn '+e.turn+' · '+(e.phase<state.boards.length?'Step 1':'Step 2')+' · '+(e.source==='human'?'You':'AI '+e.actor)+' · '+(e.action?names[e.action[0]]+' '+e.action[1]:'Passed')+' · whites '+e.white_total).join('\\n'):'No moves recorded yet';}
 const choices=state.legal.map(a=>{const c=a[0],n=a[1],details=state.explanations.find(x=>x.move===colors[c]+' '+n),sources=whitePhase?'white '+state.dice[0]+' + white '+state.dice[1]:[0,1].filter(w=>state.dice[w]+state.dice[c+2]===n).map(w=>'white '+(w+1)+' ('+state.dice[w]+') + '+colors[c]+' ('+state.dice[c+2]+')').join(' OR ');return '<li><strong>'+names[c]+' '+n+(a[2]?' 🔒':'')+'</strong> — '+sources+(details?' · +'+details.immediate+' immediate points · skips '+details.skipped:'')+'</li>'}).join('');
 if(state.done){showGameOver();if(!document.getElementById('career').hidden)showStats()}
 document.getElementById('actions').innerHTML=state.done?'<b>Game over. Start a new game.</b>':'<p>Select a highlighted number on your scorecard, then press Mark selected number under the dice, or Skip opportunity.</p><details class="move-analysis"><summary>Move analysis — '+state.legal.length+' available scorecard moves</summary><ul>'+choices+'</ul><p>Skip this opportunity with the button under the dice. Equivalent dice combinations leading to the same number are shown together.</p></details>';

}

loadPlayers().then(reset).catch(e=>showGameError(e.message));</script><details id="rules-faq"><summary>Qwixx rules & FAQ</summary>
<h3>White dice (everyone)</h3><p>Every player may mark one number matching the sum of the two white dice, or pass. These decisions are simultaneous, including row locks.</p>
<h3>Colored die (active player only)</h3><p>The active player may combine either white die with one available colored die and mark that row, or pass.</p>
<h3>What does left to right mean?</h3><p>Numbers must be marked strictly from left to right. Skipped numbers are permanently unavailable.</p>
<h3>What is a misthrow penalty?</h3><p>Only the active player takes a misthrow penalty when neither phase produces a mark. Each penalty costs five points.</p>
<h3>How does the padlock work?</h3><p>After five prior marks in a row, marking its far-right number also marks the padlock, closes that color to everyone and removes the colored die. Multiple players can lock during the same white-dice phase.</p>
<h3>When does the game end?</h3><p>After the second color is locked or any player receives their fourth misthrow penalty. Resolve simultaneous white-dice choices first.</p>
<h3>How is scoring calculated?</h3><p>Each row scores triangular points based on its marks, including the padlock bonus mark. Subtract five points per penalty. The highest score wins; ties share the win.</p>
</details></body></html>'''

HTML = HTML.replace('id="app-version" style="font-size:.48em;color:#94a3b8;font-weight:600"></small>', 'id="app-version" style="font-size:.48em;color:#94a3b8;font-weight:600">'+html.escape(GIT_VERSION)+'</small>')

class Session:
    def __init__(self,model=None,difficulty='champion'):
        if difficulty not in ('beginner','intermediate','advanced','champion'):raise ValueError('Invalid difficulty')
        self.model=model;self.difficulty=difficulty;self.lock=threading.Lock();self.game=Game();self.history=[];self.step1_snapshot=None
    def ai(self,g):
        if self.difficulty=='beginner':return random_agent(g)
        if self.difficulty=='intermediate':return heuristic(g)
        if self.difficulty=='champion':return champion(g)
        if self.model is None:return heuristic_v2(g)
        return choose(self.model,g,epsilon=0.18)
    def apply_action(self,action,source):
        g=self.game
        if g.done:raise ValueError('Game already finished')
        actor=g.actor
        if source=='human' and actor!=0:raise ValueError('Not the human player turn')
        if source=='ai' and actor==0:raise ValueError('AI cannot act for human')
        entry={'turn':g.turn+1,'phase':g.phase,'actor':actor,
               'roller':g.active,'source':source,'action':list(action) if action is not None else None,
               'white_total':g.dice[0]+g.dice[1]}
        if source=='human' and g.active==0 and g.phase==0:
            self.step1_snapshot=(copy.deepcopy(g),copy.deepcopy(self.history))
        elif source=='human' and g.active==0 and g.phase==g.n:
            self.step1_snapshot=None
        g.step(action)
        self.history.append(entry)
        if len(self.history)>200:self.history=self.history[-200:]
    def undo_step1(self):
        if not self.can_undo_step1():raise ValueError('Step 1 cannot be undone now')
        self.game,self.history=self.step1_snapshot
        self.step1_snapshot=None

    def can_undo_step1(self):
        g=self.game
        return (self.step1_snapshot is not None and not g.done and
                g.active==0 and g.phase==g.n and
                self.step1_snapshot[0].turn==g.turn)

    def advance(self):
        while not self.game.done and self.game.actor!=0:
            self.apply_action(self.ai(self.game),'ai')
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
        return {'turn':g.turn,'phase':g.phase,'actor':g.actor,'roller':g.active,'active_marks':g.active_marks,'can_undo_step1':self.can_undo_step1(),'dice':g.dice,'locks':[COLORS[c] for c in sorted(g.locks)],'done':g.done,'scores':g.scores(),'boards':[{'rows':b.rows,'penalties':b.penalties} for b in g.boards],'legal':g.legal() if not g.done else [],'difficulty':self.difficulty,'action_history':self.history[-25:],'explanations':self.explanations() if not g.done else []}

def serve(host='127.0.0.1',port=8765,model_path=None):
    session=Session(load(model_path) if model_path else None)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/training':
                body=b'Training dashboard requires the distributed server: python -m qwixx_ai.distributed'
                self.send_response(503);self.send_header('Content-Type','text/plain; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            if self.path!='/':self.send_error(404);return
            body=HTML.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if self.path not in ('/reset','/move','/undo-step1'):self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length>10000:raise ValueError('Payload too large')
                data=json.loads(self.rfile.read(length) or b'{}')
                with session.lock:
                    if self.path=='/reset':
                        difficulty=data.get('difficulty','champion')
                        if difficulty not in ('beginner','intermediate','advanced','champion'):raise ValueError('Invalid difficulty')
                        session.difficulty=difficulty;session.game=Game();session.history=[];session.step1_snapshot=None;session.advance()
                    elif self.path=='/undo-step1':
                        if data.get('expected_turn')!=session.game.turn or data.get('expected_phase')!=session.game.phase:raise ValueError('Game state changed; reload')
                        session.undo_step1()
                    else:
                        a=data.get('action');a=None if a is None else tuple(a)
                        session.apply_action(a,'human');session.advance()
                    result=session.view()
                body=json.dumps(result).encode();status=200
            except (ValueError,RuntimeError,KeyError,TypeError) as e:
                body=json.dumps({'error':str(e)}).encode();status=400
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    print(f'Open http://{host}:{port}');ThreadingHTTPServer((host,port),Handler).serve_forever()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);p.add_argument('--model');a=p.parse_args();serve(a.host,a.port,a.model)
