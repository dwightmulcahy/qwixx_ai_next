"""Local play-versus-AI web UI, standard-library HTTP server."""
import argparse,json,threading,os,html,copy
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from .core import Game,COLORS,ORDER
from .agents import heuristic,random_agent
from .model import choose,load
from .champion import champion
from .heuristic_v2 import heuristic_v2
from .favicon import FAVICON_SVG

# Supplied by the image build; no Git repository is required at runtime.
GIT_VERSION = os.getenv('QWIXX_GIT_VERSION', '').strip()

HTML='''<!doctype html><html><head><meta charset="utf-8"><title>Qwixx AI</title><link rel="icon" href="/favicon.svg" type="image/svg+xml"><style>

body{font:16px system-ui;background:#111827;color:#f9fafb;max-width:1280px;margin:24px auto;padding:0 16px}
h1{margin-bottom:4px}.hint{color:#9ca3af}button{cursor:pointer;background:#374151;color:white;border:1px solid #6b7280;border-radius:7px;padding:9px 14px;margin:4px}button:hover{background:#4b5563}button.legal{background:#2563eb}a{color:#93c5fd}
details{background:#1f2937;border-radius:8px;padding:14px;margin:24px 0}summary{cursor:pointer}
.dice-tray{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0 22px}.die-wrap{display:flex;flex-direction:column;align-items:center;gap:5px}.die-label{font-size:12px;color:#cbd5e1;font-weight:700}.die{height:66px;width:66px;border-radius:13px;display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,1fr);padding:9px;box-sizing:border-box;box-shadow:inset -5px -6px 8px #0003,0 4px 8px #0005;border:2px solid #ffffff55}.pip{width:11px;height:11px;border-radius:50%;background:#fff;align-self:center;justify-self:center;box-shadow:0 1px 2px #0005}.die.white{background:#f9fafb}.die.white .pip{background:#111827}.die.red{background:#e53e48}.die.yellow{background:#f2bb20}.die.green{background:#35a360}.die.blue{background:#347cda}.die.off{opacity:.65;filter:saturate(.55) brightness(.88)}.die.off .pip{opacity:.7}.die{position:relative}.die.locked-die:after{content:'🔒';position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:34px;background:#11182788;border-radius:11px}.game-over-overlay{position:fixed;inset:0;z-index:1000;background:#000a;display:flex;align-items:center;justify-content:center;padding:16px}.game-over-dialog{background:#1f2937;border:2px solid #94a3b8;border-radius:16px;box-shadow:0 20px 70px #0009;padding:24px;max-width:420px;width:100%;text-align:center}.game-over-dialog h2{margin-top:0}.game-over-dialog button{background:#167c54;font-weight:800}.welcome-overlay{position:fixed;inset:0;z-index:1200;background:#020617d9;display:flex;align-items:center;justify-content:center;padding:18px}.welcome-overlay[hidden]{display:none!important}.welcome-dialog{background:var(--panel);color:var(--text);border:2px solid var(--border);border-radius:18px;box-shadow:0 24px 80px #000b;padding:28px;max-width:520px;width:100%;text-align:center}.welcome-dialog h2{font-size:28px;margin:0 0 10px}.welcome-dialog p{color:var(--muted);line-height:1.55;margin:8px 0 20px}.welcome-actions{display:grid;grid-template-columns:1fr 1fr;gap:12px}.welcome-actions button{margin:0;padding:14px 16px;font-size:16px;font-weight:800}.welcome-actions .guest{background:#374151}.welcome-actions .register{background:#167c54}.welcome-note{font-size:12px!important;margin-top:14px!important}@media(max-width:520px){.welcome-actions{grid-template-columns:1fr}}
.scorecard{width:min(100%,640px);box-sizing:border-box;background:#d1d5db;padding:8px;border-radius:10px;color:#111827;margin:10px 0 18px;overflow-x:auto}.score-row{display:grid;grid-template-columns:repeat(11,minmax(24px,1fr)) 30px;gap:2px;padding:4px;margin-bottom:3px;border-radius:6px;min-width:360px}.score-row.red{background:#d83a46;--ink:#c92f3d;--light:#ffe3e7}.score-row.yellow{background:#f4d934;--ink:#ba8400;--light:#fff9df}.score-row.green{background:#329753;--ink:#267b42;--light:#e4f6e9}.score-row.blue{background:#38568e;--ink:#304e91;--light:#e6eafa}
.number-cell,.lock-cell{margin:0;padding:0;width:100%;box-sizing:border-box;min-height:38px;border:2px solid var(--ink);border-radius:11px;background:var(--light);color:var(--ink);font-size:15px;font-weight:800;display:flex;align-items:center;justify-content:center;position:relative}.number-cell.playable{cursor:pointer;z-index:2;background:#fff!important;border:3px solid #111827;outline:4px solid #f8fafc;outline-offset:-6px;box-shadow:0 0 0 5px #22c55e,0 0 18px 5px #22c55e99,inset 0 0 0 3px #fff;transform:scale(1.04);animation:available-pulse 1.35s ease-in-out infinite}.number-cell.selected{background:#fff;color:var(--ink);outline:4px solid #111827;box-shadow:0 0 0 4px #f8fafc;animation:none}.phase-controls{background:#1f2937;border-radius:10px;padding:10px 14px;max-width:650px}.phase-controls p{margin:6px 0}.phase-controls button.confirm{background:#167c54;font-weight:800}.strike-warning{color:#fbbf24;font-weight:700}.phase-controls button:disabled{opacity:.4;cursor:default}.number-cell.playable:hover,.number-cell.playable:focus-visible{background:#fff!important;color:var(--ink);transform:scale(1.07) translateY(-1px);outline:4px solid #111827;box-shadow:0 0 0 6px #4ade80,0 0 24px 7px #4ade80bb}.number-cell.playable:disabled{cursor:wait}@keyframes available-pulse{50%{box-shadow:0 0 0 6px #4ade80,0 0 22px 6px #4ade8099,inset 0 0 0 3px #fff}}.number-cell.marked{background:var(--ink);color:#fff}.number-cell.skipped{opacity:.55}.number-cell.skipped .number-label{text-decoration:line-through;text-decoration-thickness:3px;text-decoration-color:currentColor}.number-cell.marked:after{content:'✕';font-size:25px;position:absolute;color:#fff}.number-cell.marked .number-label{opacity:.25}.lock-cell{border-radius:50%;font-size:26px}.lock-cell.locked{background:var(--ink);color:white}.score-footer{display:flex;flex-wrap:wrap;align-items:center;gap:5px;padding:5px 2px;font-size:12px}.score-footer.combined{justify-content:space-between}.score-footer .missed-throws{display:flex;align-items:center;gap:4px}.score-key{display:flex;flex-wrap:wrap;gap:3px}.score-key span{background:#fff;border:1px solid #9ca3af;border-radius:5px;padding:3px 4px;font-size:10px;text-align:center}.penalty-box{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;background:white;border:2px solid #9ca3af;border-radius:7px;font-weight:900}.penalty-box.used{color:#b91c1c}.score-totals{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-weight:800}.total-box{border:3px solid var(--ink);background:#fff;border-radius:10px;min-width:52px;padding:9px;text-align:center;color:var(--ink)}.total-box.red{--ink:#d83a46}.total-box.yellow{--ink:#c69a00}.total-box.green{--ink:#329753}.total-box.blue{--ink:#38568e}.total-box.penalty{--ink:#6b7280}.total-box.final{--ink:#111827;min-width:75px}.player-heading{margin:16px 0 6px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}.rank-badge{display:inline-flex;align-items:center;justify-content:center;border:1px solid #64748b;border-radius:999px;background:#1e293b;color:#e2e8f0;padding:3px 8px;font-size:12px;font-weight:900;white-space:nowrap}.rank-badge.you{border-color:#4ade80;background:#14532d;color:#f0fdf4}.rank-badge.leader{border-color:#fbbf24;background:#422006;color:#fef3c7}.lock-cell.lock-ready{background:#f59e0b!important;color:#1c1917!important;border-color:#fde68a!important;box-shadow:0 0 0 3px #f59e0b55;cursor:help}.player-panel.inactive{opacity:.42;filter:grayscale(.85)}.player-panel.inactive .scorecard{pointer-events:none}.player-panel.inactive .number-cell.playable{pointer-events:auto}.player-panel{max-width:640px}.turn-info{line-height:1.6}
@media(max-width:650px){body{padding:0 9px}.die{width:54px;height:54px;padding:7px}.pip{width:9px;height:9px}.dice-tray{gap:7px}.scorecard{padding:7px}.score-row{min-width:470px}.number-cell,.lock-cell{min-height:38px}}
 .career-panel{background:#1f2937;border:1px solid #64748b;border-radius:12px;padding:16px;max-width:680px;margin:12px 0}.career-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:8px}.career-stat{background:#111827;padding:10px;border-radius:8px}.career-stat b{display:block;font-size:22px}.career-panel table{width:100%;border-collapse:collapse}.career-panel td,.career-panel th{padding:6px;border-bottom:1px solid #374151;text-align:left}.career-panel input,.career-panel select{padding:8px;border-radius:6px;max-width:180px} .turn-banner{max-width:650px;border:3px solid #4ade80;background:#123a2b;color:#f0fdf4;border-radius:14px;padding:16px;margin:14px 0}.turn-banner h2{font-size:27px;margin:0 0 6px}.turn-banner p{margin:0}.turn-banner.second{border-color:#fbbf24;background:#42310d}.white-total{font-size:28px;font-weight:900;background:#fff;color:#111827;border-radius:10px;display:inline-block;padding:8px 14px;margin-top:10px}.phase-controls{border:2px solid #4ade80}.phase-controls.second{border-color:#fbbf24}.player-panel.human-active .scorecard{box-shadow:0 0 0 4px #4ade80}.player-panel.human-active .player-heading{color:#86efac}.number-cell.playable{box-shadow:0 0 0 5px #22c55e,0 0 18px 5px #22c55e99,inset 0 0 0 3px #fff}.number-cell.selected{box-shadow:0 0 0 4px #fbbf24;animation:none}@media(prefers-reduced-motion:reduce){.number-cell.playable{animation:none}}.score-row.row-locked{position:relative;background:#17212e!important;border:3px solid #e2e8f0;box-shadow:inset 0 0 0 2px #0f172a;gap:2px;padding-top:28px}.score-row.row-locked:before{content:'🔒 ROW LOCKED — NO MORE MARKS';position:absolute;top:3px;left:8px;color:#f8fafc;font-size:12px;font-weight:900;letter-spacing:.06em}.score-row.row-locked .number-cell{opacity:.68;filter:grayscale(.65)}.score-row.row-locked .number-cell.marked{opacity:1;filter:none}.score-row.row-locked .lock-cell{background:#e2e8f0;color:#0f172a;border-color:#e2e8f0}.score-row.row-locked .number-cell.playable{animation:none;box-shadow:none;cursor:default}.score-row.row-locked .row-lock-overlay{position:absolute;inset:28px 4px 3px;z-index:4;pointer-events:auto;display:flex;align-items:center;justify-content:center}.score-row.row-locked .row-lock-overlay:before{content:'';position:absolute;left:0;right:0;top:50%;height:5px;transform:translateY(-50%);background:#f8fafc;box-shadow:0 1px 3px #0f172a}.score-row.row-locked .row-lock-overlay span{position:relative;padding:4px 18px;border:2px solid #f8fafc;border-radius:6px;background:#17212e;color:#fff;font-size:16px;font-weight:950;letter-spacing:.15em;box-shadow:0 2px 6px #0f172a}.roller-header{display:flex;flex-wrap:wrap;align-items:center;gap:10px;max-width:680px;margin:14px 0 4px}.turn-count{font-size:15px;font-weight:800;color:#cbd5e1}.roller-name{font-size:clamp(19px,3vw,26px);font-weight:900;padding:6px 13px;border-radius:9px}.roller-name.human{background:#166534;color:#f0fdf4;border:2px solid #4ade80}.roller-name.opponent{background:#334155;color:#fff;border:2px solid #64748b}.roller-caption{font-size:16px;color:#cbd5e1}.turn-subtitle{margin:5px 0 10px;color:#cbd5e1}.phase-controls{border:1px solid #475569!important}.phase-controls.second{border-color:#64748b!important}.player-panel.human-active .scorecard{box-shadow:none}.player-panel.human-active .player-heading{color:inherit}.app-tabs{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0;border-bottom:1px solid #475569;padding-bottom:8px}.app-tabs button{margin:0;background:transparent;border-color:transparent;font-weight:700}.app-tabs button[aria-selected="true"]{background:#334155;border-color:#64748b}.app-panel[hidden]{display:none!important}.app-panel{padding-top:8px}.app-panel .career-panel{max-width:900px}.game-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:8px}.app-panel h2{margin-top:10px}@media(max-width:650px){.app-tabs button{font-size:13px;padding:8px}}.leaderboard-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:16px}.leaderboard-card{background:#1f2937;border:1px solid #475569;border-radius:12px;padding:12px}.leaderboard-card table{width:100%;border-collapse:collapse}.leaderboard-card th,.leaderboard-card td{padding:7px;text-align:left;border-bottom:1px solid #374151}.career-panel{max-width:none!important;width:100%;box-sizing:border-box}.career-header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}.career-header h2{margin-bottom:4px}.career-grid{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))!important;gap:12px!important}.career-stat{border:1px solid #374151;padding:16px!important}.career-stat span{color:#aebbd0;font-size:13px}.career-stat b{font-size:27px!important;margin-top:7px}.career-table-wrap{overflow-x:auto}.career-table-wrap table{width:100%;min-width:650px}.career-table-wrap th,.career-table-wrap td{padding:12px!important}.career-table-wrap tbody tr:hover{background:#334155}.career-panel h3{margin-top:28px}.score-totals{flex-wrap:nowrap;gap:clamp(2px,.65vw,7px);width:100%;box-sizing:border-box;white-space:nowrap}.score-totals>strong{flex:0 0 auto;font-size:clamp(11px,1.25vw,14px)}.score-totals>.total-box{box-sizing:border-box;flex:1 1 0;min-width:0;padding:8px 2px;border-width:2px;font-size:clamp(12px,1.65vw,18px)}.score-totals>.total-box.final{flex:1.4 1 0;min-width:0;background:#111827;color:#fff;border:3px solid #111827;font-size:clamp(20px,2.7vw,30px);font-weight:950;box-shadow:0 3px 0 #64748b}.score-totals>span:not(.total-box){flex:0 0 auto;font-size:clamp(11px,1.4vw,17px)}@media(max-width:650px){.score-totals{gap:3px}.score-totals>.total-box{padding:7px 1px}.score-totals>strong{font-size:11px}.score-totals>.total-box.final{font-size:22px}}
:root{color-scheme:dark;--page:#0b1421;--panel:#142335;--cell:#1d3045;--text:#f4f8ff;--muted:#b8c9df;--border:#36516c;--qw-red:#d83a46;--qw-yellow:#f4d934;--qw-green:#329753;--qw-blue:#38568e}
html[data-theme="light"]{color-scheme:light;--page:#f0f4f9;--panel:#fff;--cell:#e7eef6;--text:#172334;--muted:#43556d;--border:#a9bed3}
body{background:var(--page);color:var(--text);max-width:1540px}
.hint,.turn-subtitle,.roller-caption,.turn-count{color:var(--muted)}
details,.career-panel,.phase-controls,.game-over-dialog{background:var(--panel);color:var(--text);border-color:var(--border)}
.theme-picker{display:inline-flex;align-items:center;gap:7px;margin-left:auto;font-size:13px}
.theme-picker select{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px}
#status{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:10px 14px;margin:8px 0 12px}
#status .roller-header{margin:0 0 4px}#status .dice-tray{margin:8px 0 4px}
.game-grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(350px,1fr);gap:16px;align-items:start}
.human-column,.opponents-column{min-width:0}.human-column .player-panel{max-width:none}
.human-column .scorecard{width:100%;max-width:none;margin:8px 0}
.human-column .score-row{grid-template-columns:repeat(11,minmax(0,1fr)) 30px;min-width:0}
.human-column .number-cell{min-width:0}
.human-column .phase-controls{max-width:none;border:1px solid var(--border)!important}
.player-panel.inactive{opacity:1;filter:none}
.mini-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:12px;margin-bottom:10px;min-width:0}
.mini-card.rolling{border:2px solid #60a5fa}.mini-card.leading{border-color:#fbbf24}
.mini-header,.mini-footer{display:flex;justify-content:space-between;align-items:center;gap:6px}
.mini-header{font-weight:800;margin-bottom:9px}.mini-header span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.mini-header-meta{display:flex;align-items:center;gap:5px;flex:0 0 auto}.lock-threats{display:flex;flex-wrap:wrap;gap:4px;margin:-2px 0 7px}.lock-threat-badge{display:inline-flex;align-items:center;gap:4px;border:1px solid #f59e0b;border-radius:999px;background:#451a03;color:#fde68a;padding:3px 7px;font-size:10px;font-weight:900;letter-spacing:.03em;white-space:nowrap}.mini-row.lock-ready{outline:2px solid #f59e0b;outline-offset:-1px;box-shadow:0 0 10px #f59e0b66}.mini-lock.lock-ready{background:#f59e0b;color:#1c1917;border-radius:5px;font-weight:900;cursor:help}
.mini-footer{font-size:12px;color:var(--muted);margin-top:9px}
.mini-row{display:grid;grid-template-columns:repeat(11,minmax(0,1fr)) 21px;gap:2px;margin:3px 0}
.mini-number{display:flex;align-items:center;justify-content:center;min-width:0;min-height:25px;background:var(--cell);border:1px solid var(--border);border-radius:4px;font-size:clamp(10px,.85vw,13px);font-weight:750;position:relative}
.mini-number.marked{background:var(--row-color);color:#fff;border-color:var(--row-color)}
.mini-row.yellow .mini-number.marked{color:#1a2433}
.mini-number.marked:after{content:'×';font-size:10px;position:absolute;bottom:-3px;right:1px}
.mini-number.skipped{text-decoration:line-through;opacity:.65}
.mini-lock{display:flex;align-items:center;justify-content:center;font-size:14px}
.mini-row.red{--row-color:var(--qw-red)}.mini-row.yellow{--row-color:var(--qw-yellow)}.mini-row.green{--row-color:var(--qw-green)}.mini-row.blue{--row-color:var(--qw-blue)}
html[data-theme="light"] .scorecard{background:#dbe4ee}
html[data-theme="light"] .phase-controls button:not(.confirm){background:var(--cell);color:var(--text);border-color:var(--border)}
html[data-theme="light"] .app-tabs button,html[data-theme="light"] .game-toolbar button{color:var(--text)}
html[data-theme="light"] .app-tabs button[aria-selected="true"]{background:var(--cell)}
html[data-theme="light"] .career-stat{background:var(--cell)}

/* Align the dice with the human card; opponents start at the same height. */
#status{display:none}
.dice-status{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:10px 14px;margin:0 0 12px}
.dice-status .roller-header{margin:0 0 4px}.dice-status .dice-tray{margin:8px 0 4px}
.game-grid{align-items:start}
.mini-row{border-radius:5px;padding:3px;background:var(--row-color)}
.mini-row.red{--row-color:var(--qw-red)}
.mini-row.yellow{--row-color:var(--qw-yellow)}
.mini-row.green{--row-color:var(--qw-green)}
.mini-row.blue{--row-color:var(--qw-blue)}
.mini-number{background:var(--panel);color:var(--text);border-color:color-mix(in srgb,var(--row-color) 55%,var(--border))}
.mini-number.marked{background:color-mix(in srgb,var(--row-color) 78%,#101827);border-color:var(--row-color);color:#fff}
.mini-row.yellow .mini-number.marked{color:#fff}
.number-cell.skipped .number-label{text-decoration:none}
.number-cell.skipped:after,.mini-number.skipped:after{content:'';position:absolute;left:3px;right:3px;top:50%;height:3px;background:currentColor;transform:translateY(-50%);pointer-events:none;border-radius:3px;opacity:.9}
.mini-number.skipped{text-decoration:none;opacity:.8}
.mini-number.skipped:after{height:2px;left:1px;right:1px}
.mini-row-locked{position:relative;outline:2px solid #aab8cc;outline-offset:-2px;background:#17212e!important;filter:none}.mini-row-locked .mini-number{background:#17212e;color:#a6b2c3;border-color:#344255;opacity:.55}.mini-row-locked .mini-number.marked{background:#263447;color:#dbe4ef;border-color:#526174;opacity:.85}.mini-row-locked .mini-number.skipped:after{opacity:1}.mini-row-locked .mini-lock{color:#e2e8f0}.mini-row-locked:after{content:"";position:absolute;left:5px;right:24px;top:50%;height:3px;transform:translateY(-50%);background:#e2e8f0;opacity:.85;pointer-events:none;z-index:1}
.mini-locked-label{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:2;background:#101b2e;color:#fff;border:1px solid #d1dae8;border-radius:4px;padding:2px 9px;font-size:11px;font-weight:800;letter-spacing:.08em;white-space:nowrap;pointer-events:auto;cursor:help;box-shadow:0 1px 5px #0008}
@media(max-width:980px){.game-grid{grid-template-columns:1fr}.opponents-column{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.mini-card{padding:7px}.mini-row{grid-template-columns:repeat(11,minmax(0,1fr)) 15px;gap:1px}.mini-number{font-size:9px;min-height:21px}.mini-header{font-size:12px}}
@media(max-width:640px){.opponents-column{grid-template-columns:1fr}.mini-number{font-size:12px}.human-column .score-row{min-width:470px}.theme-picker{margin-left:0}.game-toolbar{flex-wrap:wrap}}
.mini-card.ai-active{border:2px solid #60a5fa;box-shadow:0 0 0 2px #60a5fa44}.mini-card.ai-active .mini-header{color:#93c5fd}
.ai-thinking{position:absolute;top:8px;right:10px;z-index:5;display:inline-flex;align-items:center;gap:9px;padding:7px 11px;margin:0;border:1px solid #60a5fa;border-radius:9px;background:var(--panel);color:var(--text);font-weight:700;box-shadow:0 3px 12px #0006;pointer-events:none;animation:thinking-appear .15s ease .15s both}.dice-tray{align-items:center}.phase-controls button:disabled{opacity:.35;filter:grayscale(1);cursor:not-allowed}.ai-thinking-spinner{width:17px;height:17px;border:3px solid #64748b;border-top-color:#60a5fa;border-radius:50%;animation:ai-thinking-spin .75s linear infinite}@keyframes ai-thinking-spin{to{transform:rotate(360deg)}}@keyframes thinking-appear{from{opacity:0}to{opacity:1}}@media(prefers-reduced-motion:reduce){.ai-thinking,.ai-thinking-spinner{animation:none}}

/* Compact dice area: keep turn/step alongside the dice, not stacked above them. */
.dice-status{position:relative;padding:9px 12px;margin-bottom:9px}
.compact-dice-layout{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.compact-turn{flex:0 1 270px;min-width:185px}
.compact-turn .roller-header{margin:0 0 4px;gap:6px}
.compact-turn .roller-name{font-size:clamp(16px,2vw,22px);padding:4px 9px}
.compact-turn .turn-count,.compact-turn .roller-caption{font-size:14px}
.compact-turn .turn-subtitle{margin:3px 0 0;font-size:13px}
.dice-status .dice-tray{margin:0;gap:7px;flex:1 1 360px}
.dice-status .die{width:52px;height:52px;padding:6px;border-radius:11px}
.dice-status .pip{width:8px;height:8px}
.dice-status .die-label{font-size:10px}
.dice-status .die-wrap{gap:2px}
.dice-status .die.locked-die:after{font-size:26px}
.mini-card.ai-active,.mini-card.rolling{border:3px solid #39e6ba!important;background:#1b3546;box-shadow:0 0 0 2px #39e6ba66,0 0 22px #39e6ba40}
.mini-card.ai-active .mini-header,.mini-card.rolling .mini-header{color:#f0fffc;background:#22516a;border-radius:8px;padding:5px 7px}
.mini-card.ai-inactive{border-color:#526174;opacity:.82}
.active-ai-badge{display:inline-block;background:#087b63;color:#fff;border:1px solid #7affdc;border-radius:7px;padding:3px 7px;font-size:11px;white-space:nowrap;vertical-align:middle}
@media(max-width:700px){.compact-turn{flex:1 1 100%}.dice-status .dice-tray{flex-basis:100%}.dice-status .die{width:46px;height:46px}.ai-thinking{top:6px;right:7px;padding:5px 7px}.ai-thinking>span:last-child{display:none}}
/* Keep the current move controls inside the dice tile so the active-turn UI stays together. */
.dice-move-controls{border-top:1px solid var(--border);margin-top:9px;padding-top:9px;display:grid;grid-template-columns:minmax(0,1fr) auto;grid-template-areas:"status buttons" "hint shortcuts";gap:5px 12px;align-items:center}
.dice-move-controls .move-status{grid-area:status;display:flex;flex-wrap:wrap;gap:4px 12px;align-items:center;min-width:0}
.dice-move-controls .phase-instruction{font-size:15px;font-weight:750}
.dice-move-controls .selected-mark{font-size:12px;color:var(--muted);white-space:nowrap}
.dice-move-controls .move-hint{grid-area:hint;font-size:12px;color:var(--muted);min-height:17px}
.dice-move-controls .move-hint.strike-warning{color:#fbbf24}
.dice-move-controls .move-buttons{grid-area:buttons;display:flex;align-items:center;gap:6px;flex-wrap:nowrap}
.dice-move-controls .move-buttons button{margin:0;padding:8px 13px;font-size:14px;white-space:nowrap;min-height:38px}
.dice-move-controls .move-buttons .confirm{min-width:86px}
.dice-shortcuts{grid-area:shortcuts;text-align:right;color:var(--muted);font-size:11px;white-space:nowrap}
.auto-pass-status{grid-area:buttons;background:#0f766e;border:1px solid #5eead4;color:#ecfeff;border-radius:9px;padding:9px 12px;font-size:13px;font-weight:900;letter-spacing:.04em;white-space:nowrap}
@media(max-width:760px){.dice-move-controls{grid-template-columns:1fr;grid-template-areas:"status" "buttons" "hint" "shortcuts"}.dice-move-controls .move-buttons{flex-wrap:wrap}.dice-move-controls .move-buttons button{flex:1 1 auto}.dice-shortcuts{text-align:left;white-space:normal}.auto-pass-status{white-space:normal}}

/* Gameplay clarity: highlight changed AI cells and keep a small nonblocking event strip. */
.mini-number.recent-mark,.number-cell.recent-mark{outline:3px solid #fef08a;outline-offset:1px;box-shadow:0 0 12px #facc15}
.mini-row.recent-lock{box-shadow:inset 0 0 0 3px #facc15}
.game-events{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:5px 0;font-size:12px;min-height:22px}
.game-event{padding:3px 8px;border-radius:6px;background:#164e63;color:#ecfeff}
.game-event.strike{background:#78350f;color:#fef3c7}
.game-event.lock{background:#365314;color:#f7fee7}
.next-roller{font-size:12px;color:var(--muted);margin-left:8px}
@media(prefers-reduced-motion:reduce){.mini-number.recent-mark,.number-cell.recent-mark{transition:none}}

.move-analysis-panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:10px 12px;margin:10px 0 4px}.move-analysis-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.move-analysis-actions button{padding:6px 10px}.move-analysis-panel .move-assist{margin:7px 0}.move-analysis{margin:7px 0 0;padding:8px 10px}.move-analysis summary{font-weight:750}.gameplay-overlay{position:fixed;inset:0;z-index:1300;background:#020617d9;display:flex;align-items:center;justify-content:center;padding:18px}.gameplay-overlay[hidden]{display:none!important}.gameplay-dialog{background:var(--panel);color:var(--text);border:2px solid var(--border);border-radius:16px;box-shadow:0 24px 80px #000b;padding:22px;max-width:520px;width:100%}.gameplay-dialog-header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px}.gameplay-dialog-header h2{margin:0}.gameplay-dialog-header button{margin:0;padding:6px 10px}.gameplay-pref-grid{display:grid;gap:10px}.gameplay-pref-row{display:flex;justify-content:space-between;align-items:center;gap:18px;padding:10px 0;border-bottom:1px solid var(--border)}.gameplay-pref-row:last-child{border-bottom:0}.gameplay-pref-row span{display:block;color:var(--muted);font-size:12px;margin-top:2px}.gameplay-pref-row select{min-width:130px}.gameplay-button{margin-left:auto}

html[data-reduced-motion="on"] .ai-thinking-spinner,html[data-reduced-motion="on"] .number-cell.playable{animation:none!important;transition:none!important}.move-assist{font-size:12px;color:var(--muted);margin:4px 0}
</style></head><body><h1>Qwixx AI <small id="app-version" style="font-size:.48em;color:#94a3b8;font-weight:600"></small></h1><div id="welcome-modal" class="welcome-overlay" role="dialog" aria-modal="true" aria-labelledby="welcome-title" hidden><div class="welcome-dialog"><h2 id="welcome-title">Welcome to Qwixx AI</h2><p id="welcome-copy">Choose how you want to play this session.</p><div class="welcome-actions"><button type="button" class="guest" onclick="chooseGuestPlay()">Guest Play</button><button type="button" id="welcome-primary" class="register" onclick="choosePrimaryPlay()">Register / Sign in</button></div><p class="welcome-note">Guest play lasts only for this browser session. You will choose again on your next visit.</p></div></div><div id="gameplay-modal" class="gameplay-overlay" role="dialog" aria-modal="true" aria-labelledby="gameplay-title" hidden onclick="if(event.target===this)closeGameplayPreferences()"><div class="gameplay-dialog"><div class="gameplay-dialog-header"><h2 id="gameplay-title">Gameplay preferences</h2><button type="button" onclick="closeGameplayPreferences()" aria-label="Close gameplay preferences">✕</button></div><div class="gameplay-pref-grid"><label class="gameplay-pref-row"><div><strong>Quick Mark</strong><span>Immediately submit a legal mark when you click it.</span></div><input id="pref-quick-mark" type="checkbox" onchange="setQuickMark(this.checked)"></label><label class="gameplay-pref-row"><div><strong>AI pace</strong><span>Controls how long AI move feedback stays visible.</span></div><select id="pref-ai-pace" onchange="setPreference(SPEED_KEY,this.value)"><option value="fast">Fast</option><option value="normal">Normal</option><option value="relaxed">Relaxed</option></select></label><label class="gameplay-pref-row"><div><strong>Motion</strong><span>Controls animated move and thinking feedback.</span></div><select id="pref-motion" onchange="setPreference(MOTION_KEY,this.value)"><option value="system">System</option><option value="on">On</option><option value="off">Off</option></select></label><label class="gameplay-pref-row"><div><strong>Move hints</strong><span>Show the best immediate-point options below your scorecard.</span></div><input id="pref-move-hints" type="checkbox" onchange="setPreference(ASSIST_KEY,this.checked?&quot;on&quot;:&quot;off&quot;)"></label></div></div></div><p><a href="/training">Training dashboard</a> · <a href="#rules-faq">Rules & FAQ</a></p><p class="hint">You play seat 1 against three AI opponents. Click a highlighted number on your scorecard, or skip a phase.</p>
<label for="difficulty">AI difficulty</label><select id="difficulty"><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option><option value="champion" selected>Champion</option></select><button onclick="reset()">New game</button> <span id="account-status">Guest</span> <button onclick="toggleStats()">Your Stats</button> <button id="auth-logout" onclick="logout()" hidden>Sign out</button><div id="auth-login"><label>Email <input id="auth-email" type="email" placeholder="you@example.com"></label><label> Name (new accounts) <input id="auth-name" placeholder="Display name"></label><button onclick="requestCode()">Email me a code</button><div id="verify-box" hidden><label>Six-digit code <input id="auth-code" inputmode="numeric" maxlength="6"></label><button onclick="verifyCode()">Verify & sign in</button></div><p id="auth-message"></p></div><section id="career" class="career-panel" hidden></section><p id="game-error" role="alert" style="background:#7f1d1d;padding:12px;border-radius:8px" hidden></p><div id="status"><p>Loading game…</p></div><div id="boards"></div><details><summary>Turn action history (diagnostics)</summary><pre id="game-history" style="white-space:pre-wrap"></pre></details>
<script>
(function(){
const sections=[['play','Play'],['account','Account & Stats'],['leaderboard','Top 10'],['training','Training'],['rules','Rules & FAQ']];
function activate(id){
 for(const [key] of sections){const panel=document.getElementById('section-'+key),tab=document.getElementById('tab-'+key);if(!panel||!tab)continue;panel.hidden=key!==id;tab.setAttribute('aria-selected',String(key===id));tab.tabIndex=key===id?0:-1}
 if(id==='account'&&typeof showStats==='function')showStats();
 if(id==='leaderboard'&&typeof showLeaderboard==='function')showLeaderboard();
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
 const themeLabel=document.createElement('label');themeLabel.className='theme-picker';themeLabel.htmlFor='theme-select';themeLabel.textContent='Appearance';
 const themeSelect=document.createElement('select');themeSelect.id='theme-select';themeSelect.setAttribute('aria-label','Appearance theme');
 for(const [value,label] of [['dark','Dark'],['light','Light'],['auto','Auto (system)']]){const option=document.createElement('option');option.value=value;option.textContent=label;themeSelect.appendChild(option)}
 themeSelect.value=readTheme();themeSelect.addEventListener('change',()=>setTheme(themeSelect.value));themeLabel.appendChild(themeSelect);toolbar.appendChild(themeLabel);const gameplayButton=document.createElement('button');gameplayButton.type='button';gameplayButton.className='gameplay-button';gameplayButton.textContent='⚙ Gameplay';gameplayButton.addEventListener('click',openGameplayPreferences);toolbar.appendChild(gameplayButton);
 for(const id of ['account-status','auth-logout','auth-login','career'])move(id,'account');
 const leader=document.createElement('div');leader.id='leaderboard';panels.leaderboard.appendChild(leader);
 const statsButton=document.querySelector('button[onclick="toggleStats()"]');if(statsButton)statsButton.remove();
 for(const id of ['game-error','status','boards'])move(id,'play');
 const history=document.getElementById('game-history');if(history&&history.closest('details'))panels.play.appendChild(history.closest('details'));
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
const THEME_KEY='qwixx-theme';
function readTheme(){try{const v=localStorage.getItem(THEME_KEY);return ['dark','light','auto'].includes(v)?v:'dark'}catch(e){return 'dark'}}
function applyTheme(v){const light=v==='light'||(v==='auto'&&window.matchMedia&&window.matchMedia('(prefers-color-scheme: light)').matches);document.documentElement.dataset.theme=light?'light':'dark'}
function setTheme(v){if(!['dark','light','auto'].includes(v))return;try{localStorage.setItem(THEME_KEY,v)}catch(e){}applyTheme(v)}
applyTheme(readTheme());
if(window.matchMedia){const media=window.matchMedia('(prefers-color-scheme: light)');if(media.addEventListener)media.addEventListener('change',()=>applyTheme(readTheme()))}
let state, pending=null, gameOverShown=false, thinkingActor=null, autoPassTimer=null, autoPassKey='';
const QUICK_MARK_KEY='qwixx-quick-mark';
const SPEED_KEY='qwixx-ai-speed',ASSIST_KEY='qwixx-move-assist',MOTION_KEY='qwixx-reduced-motion';
const PLAY_MODE_KEY='qwixx-play-mode';
function preference(key,fallback){try{return localStorage.getItem(key)||fallback}catch(e){return fallback}}
function setPreference(key,value){try{localStorage.setItem(key,value)}catch(e){}render()}
function eventDuration(){return ({fast:450,normal:1100,relaxed:2200})[preference(SPEED_KEY,'normal')]||1100}
function reducedMotion(){return preference(MOTION_KEY,'system')==='on'||(preference(MOTION_KEY,'system')==='system'&&window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches)}
function syncGameplayPreferences(){const q=document.getElementById('pref-quick-mark'),pace=document.getElementById('pref-ai-pace'),motion=document.getElementById('pref-motion'),hints=document.getElementById('pref-move-hints');if(q)q.checked=quickMarkEnabled();if(pace)pace.value=preference(SPEED_KEY,'normal');if(motion)motion.value=preference(MOTION_KEY,'system');if(hints)hints.checked=preference(ASSIST_KEY,'off')==='on'}
function openGameplayPreferences(){syncGameplayPreferences();const modal=document.getElementById('gameplay-modal');if(modal){modal.hidden=false;modal.querySelector('button')?.focus()}}
function closeGameplayPreferences(){const modal=document.getElementById('gameplay-modal');if(modal)modal.hidden=true}
function playMode(){try{return sessionStorage.getItem(PLAY_MODE_KEY)||''}catch(e){return ''}}
function setPlayMode(mode){try{sessionStorage.setItem(PLAY_MODE_KEY,mode)}catch(e){}}
function clearPlayMode(){try{sessionStorage.removeItem(PLAY_MODE_KEY)}catch(e){}}
function hideWelcome(){const modal=document.getElementById('welcome-modal');if(modal)modal.hidden=true}
function maybeShowWelcome(){const modal=document.getElementById('welcome-modal');if(!modal||!authAvailable){hideWelcome();return false}const mode=playMode();if(mode==='guest'||(mode==='registered'&&verifiedPlayer)){hideWelcome();return false}const copy=document.getElementById('welcome-copy'),primary=document.getElementById('welcome-primary');if(verifiedPlayer){if(copy)copy.textContent='Continue as your registered user to save this game and update your stats, or play this browser session as a guest.';if(primary)primary.textContent='Registered User — '+verifiedPlayer.name}else{if(copy)copy.textContent='Play this browser session as a guest, or register/sign in to save games, track statistics, and appear on the leaderboards.';if(primary)primary.textContent='Register / Sign in'}modal.hidden=false;const guest=modal.querySelector('.guest');if(guest)guest.focus();return true}
function chooseGuestPlay(){setPlayMode('guest');hideWelcome();updateAccountStatus();reset()}
function choosePrimaryPlay(){if(verifiedPlayer){setPlayMode('registered');hideWelcome();updateAccountStatus();reset();return}hideWelcome();if(typeof showTab==='function')showTab('account');setTimeout(()=>document.getElementById('auth-email')?.focus(),0)}

function quickMarkEnabled(){try{return localStorage.getItem(QUICK_MARK_KEY)==='true'}catch(e){return false}}
function setQuickMark(value){try{localStorage.setItem(QUICK_MARK_KEY,value?'true':'false')}catch(e){}render()}
function focusLegalMoves(){const first=document.querySelector('.number-cell.playable');if(first){first.scrollIntoView({behavior:reducedMotion()?'auto':'smooth',block:'center'});first.focus({preventScroll:true})}}
function keyboardMove(e){if(e.defaultPrevented||e.ctrlKey&&e.key.toLowerCase()!=='z'||e.altKey||e.metaKey&&e.key.toLowerCase()!=='z')return;
 const target=e.target;if(target&&(target.isContentEditable||/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)))return;
 if(document.querySelector('[role="dialog"]:not([hidden]), [role="alertdialog"]:not([hidden])')){if(e.key==='Escape'&&!document.getElementById('gameplay-modal')?.hidden){e.preventDefault();closeGameplayPreferences()}return;}
 if(!state||state.done||moving)return;
 if(e.key==='Enter'&&pending){e.preventDefault();confirmPhase()}
 else if(e.key==='Escape'&&pending){e.preventDefault();clearSelection()}
 else if(e.code==='Space'){e.preventDefault();skipPhase()}
 else if(e.key.toLowerCase()==='z'&&(e.ctrlKey||e.metaKey)&&state.can_undo_step1){e.preventDefault();backToStep1()}
 else if(e.key.toLowerCase()==='n'&&!e.ctrlKey&&!e.metaKey){e.preventDefault();if(window.confirm('Start a new game?'))reset()}
}
if(typeof document.addEventListener==='function')document.addEventListener('keydown',keyboardMove);

let recentChanges=[],recentTimer=null;
function recordMoveChanges(previous,next){
 if(!previous||!next||!previous.boards||!next.boards)return;
 const events=[];
 next.boards.forEach((board,i)=>{
  board.rows.forEach((row,c)=>{
   const before=previous.boards[i]?.rows[c]||[];
   row.filter(n=>!before.includes(n)).forEach(n=>events.push({player:i,color:c,number:n,type:'mark',label:playerNames[i]+' marked '+names[c]+' '+n}));
  });
  const strikes=board.penalties-(previous.boards[i]?.penalties||0);
  if(strikes>0)events.push({player:i,type:'strike',label:playerNames[i]+' received '+strikes+' strike'+(strikes>1?'s':'')});
 });
 (next.locks||[]).filter(color=>!(previous.locks||[]).includes(color)).forEach(color=>events.push({type:'lock',color:colors.indexOf(color),label:names[colors.indexOf(color)]+' row locked'}));
 recentChanges=events;
 if(recentTimer)clearTimeout(recentTimer);
 recentTimer=setTimeout(()=>{recentChanges=[];if(state)render()},eventDuration());
}
function recentMark(i,c,n){return recentChanges.some(e=>e.type==='mark'&&e.player===i&&e.color===c&&e.number===n)}


async function api(path,body){try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});const j=await r.json();if(!r.ok)throw Error(j.error||'Request failed ('+r.status+')');return j}catch(e){if(path.startsWith('/api/auth/')){const msg=document.getElementById('auth-message');if(msg)msg.textContent='Email verification error: '+e.message+'. Check SMTP settings and container logs.';console.error('Qwixx authentication:',e)}else showGameError(e.message);return null}}
function showGameError(message){const el=document.getElementById('game-error');if(el){el.hidden=false;el.textContent='Game error: '+message+' — please try New game or reload the page.'}console.error('Qwixx:',message)}
async function reset(){if(autoPassTimer){clearTimeout(autoPassTimer);autoPassTimer=null}autoPassKey='';const next=await api('/reset',{difficulty:document.getElementById('difficulty').value,guest:playMode()==='guest'});if(next){document.getElementById('game-error').hidden=true;state=next;pending=null;recentChanges=[];gameOverShown=false;document.getElementById('game-over-modal')?.remove();render()}}
let verifiedPlayer=null,verificationEmail='',authAvailable=false;
function updateAccountStatus(){const el=document.getElementById('account-status');if(!el)return;el.textContent=verifiedPlayer?'Signed in: '+verifiedPlayer.name+(playMode()==='guest'?' · playing this session as Guest':''):'Guest — games will not be saved'}
async function loadPlayers(){try{const response=await fetch('/api/auth/me');if(!response.ok)throw Error('Account status unavailable');authAvailable=true;const data=await response.json();verifiedPlayer=data.player;updateAccountStatus();document.getElementById('auth-login').hidden=!!verifiedPlayer;document.getElementById('auth-logout').hidden=!verifiedPlayer;return true}catch(e){console.warn('Account unavailable',e);return false}}
async function requestCode(){document.getElementById('auth-message').textContent='Requesting verification email…';const email=document.getElementById('auth-email').value.trim();const name=document.getElementById('auth-name').value.trim();const response=await api('/api/auth/request',{email,name:name||null});if(response){verificationEmail=email;document.getElementById('verify-box').hidden=false;document.getElementById('auth-message').textContent='Code sent. Check your email (expires in 10 minutes).'}}
async function verifyCode(){const code=document.getElementById('auth-code').value.trim();const response=await api('/api/auth/verify',{email:verificationEmail,code});if(response){document.getElementById('auth-message').textContent='Email verified. Start a new game to track stats.';document.getElementById('verify-box').hidden=true;await loadPlayers();setPlayMode('registered');hideWelcome();await reset();if(!document.getElementById('career').hidden)showStats()}}
async function logout(){await api('/api/auth/logout',{});verifiedPlayer=null;clearPlayMode();await loadPlayers();document.getElementById('career').hidden=true;maybeShowWelcome()}
function toggleStats(){const panel=document.getElementById('career');panel.hidden=!panel.hidden;if(!panel.hidden)showStats()}
async function showStats(){
 const panel=document.getElementById('career');panel.hidden=false;
 if(!verifiedPlayer){panel.innerHTML='<h2>Your game statistics</h2><p>Sign in with your email to track your completed games and view your results.</p>';return}
 panel.innerHTML='<p class="hint">Loading your game statistics…</p>';
 try{
  const response=await fetch('/api/player-stats?limit=100',{credentials:'same-origin'});
  const data=await response.json();
  if(!response.ok)throw Error(data.error||'Unable to load player stats (HTTP '+response.status+')');
  const fmt=x=>x===null||x===undefined?'—':escapeHtml(x);
  const stat=(name,value)=>'<div class="career-stat"><span>'+escapeHtml(name)+'</span><b>'+fmt(value)+'</b></div>';
  const fields=[['Games played',data.games],['Wins',data.wins],['Losses',data.losses],['Ties',data.ties],['Win rate',data.win_rate+'%'],['Average score',data.average_score],['Highest score',data.highest_score],['Lowest score',data.lowest_score],['Average penalties',data.average_penalties],['Average turns',data.average_turns],['Current win streak',data.current_win_streak],['Longest win streak',data.longest_win_streak]];
  const levels=['beginner','intermediate','advanced','champion'];
  const difficulties=levels.map(level=>{const d=data.by_difficulty[level];return '<tr><td>'+escapeHtml(level)+'</td><td>'+(d?d.games:0)+'</td><td>'+(d?d.wins:0)+'</td><td>'+(d?Math.round(d.wins/d.games*1000)/10+'%':'—')+'</td><td>'+(d?d.average_score:'—')+'</td></tr>'}).join('');
  const recent=data.recent||[];
  const date=t=>{if(!t)return '—';const d=new Date(Number(t)*1000);return Number.isNaN(d.getTime())?'—':escapeHtml(d.toLocaleString())};
  const history=recent.map(g=>'<tr><td>'+date(g.created)+'</td><td>'+fmt(g.difficulty)+'</td><td>'+fmt(g.result)+'</td><td>'+fmt(g.score)+'</td><td>'+fmt(g.penalties)+'</td><td>'+fmt(g.turns)+'</td><td>'+((g.scores||[]).map(fmt).join(' / '))+'</td></tr>').join('');
  panel.innerHTML='<div class="career-header"><div><h2>'+escapeHtml(data.player.name)+' — career statistics</h2><p class="hint">Your completed, signed-in games</p></div><button type="button" onclick="showStats()">↻ Refresh</button></div><div class="career-grid">'+fields.map(([n,v])=>stat(n,v)).join('')+'</div><h3>Results by AI difficulty</h3><div class="career-table-wrap"><table><thead><tr><th>Difficulty</th><th>Games</th><th>Wins</th><th>Win rate</th><th>Average score</th></tr></thead><tbody>'+difficulties+'</tbody></table></div><h3>Recent completed games <small>(latest '+recent.length+')</small></h3>'+(recent.length?'<div class="career-table-wrap"><table><thead><tr><th>Date</th><th>Difficulty</th><th>Result</th><th>Your score</th><th>Penalties</th><th>Turns</th><th>All player scores (You / AI 1 / AI 2 / AI 3)</th></tr></thead><tbody>'+history+'</tbody></table></div>':'<p class="hint">No completed games recorded yet. Finish a signed-in game to start tracking your results.</p>')+'<p class="hint">Showing up to 100 recent games. Career totals include all saved games.</p>';
 }catch(e){panel.innerHTML='<h2>Player statistics</h2><p role="alert">Unable to load stats: '+escapeHtml(e.message)+'</p><button type="button" onclick="showStats()">Try again</button>';console.error('Player stats:',e)}
}
async function showLeaderboard(){
 const panel=document.getElementById('leaderboard');if(!panel)return;
 const select=document.getElementById('leaderboard-difficulty');const difficulty=select?select.value:'champion';
 panel.innerHTML='<h2>Top 10 · Registered Players</h2><p>Loading rankings…</p>';
 try{
  const response=await fetch('/api/leaderboard'+(difficulty?'?difficulty='+encodeURIComponent(difficulty):''));
  if(!response.ok)throw Error('Leaderboard unavailable (HTTP '+response.status+')');
  const data=await response.json();
  const metric=[['most_wins','Most wins','wins'],['highest_score','Highest single-game score','highest_score'],['total_points','Most total points','total_points'],['average_score','Best average score','average_score'],['win_rate','Best win rate','win_rate'],['most_games','Most games played','games']];
  const picker='<label for="leaderboard-difficulty">Separate leaderboard: </label><select id="leaderboard-difficulty" onchange="showLeaderboard()">'+[['champion','🏆 Champion'],['advanced','Advanced'],['intermediate','Intermediate'],['beginner','Beginner']].map(([v,label])=>'<option value="'+v+'"'+(v===difficulty?' selected':'')+'>'+label+'</option>').join('')+'</select>';
  const tables=metric.map(([key,label,field])=>{
   const rows=data.rankings[key]||[];
   return '<section class="leaderboard-card"><h3>'+label+'</h3>'+(rows.length?'<div class="career-table-wrap"><table><thead><tr><th>#</th><th>Player</th><th>'+label+'</th><th>Games</th><th>Wins</th></tr></thead><tbody>'+rows.map((p,i)=>'<tr><td>'+(i+1)+'</td><td>'+escapeHtml(p.name)+'</td><td>'+escapeHtml(p[field])+(field==='win_rate'?'%':'')+'</td><td>'+p.games+'</td><td>'+p.wins+'</td></tr>').join('')+'</tbody></table></div>':'<p class="hint">No qualifying players yet.</p>')+'</section>';
  }).join('');
  panel.innerHTML='<h2>🏆 Top 10 · '+escapeHtml(difficulty.charAt(0).toUpperCase()+difficulty.slice(1))+' AI</h2><p class="hint">Each AI difficulty has its own rankings; results never mix across difficulty levels. Only completed games played while signed in count. Rankings show public display names, never email addresses. Average score and win rate require at least '+data.minimum_games_for_rates+' games.</p>'+picker+'<p class="hint">'+data.registered_players+' registered players with results</p><div class="leaderboard-grid">'+tables+'</div>';
 }catch(e){panel.innerHTML='<h2>Top 10</h2><p role="alert">'+escapeHtml(e.message)+'</p><button type="button" onclick="showLeaderboard()">Retry</button>'}
}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
let moving=false;function showNewPenalties(previous,next){
 if(!previous||!next||!previous.boards||!next.boards)return;
 const changes=next.boards.flatMap((board,i)=>{
  const before=previous.boards[i]?.penalties||0;
  return board.penalties>before?[{name:playerNames[i]||'Player '+(i+1),count:board.penalties-before}]:[];
 });
 if(!changes.length)return;
 document.getElementById('penalty-modal')?.remove();
 const overlay=document.createElement('div');
 overlay.id='penalty-modal';overlay.className='game-over-overlay';
 overlay.setAttribute('role','alertdialog');overlay.setAttribute('aria-modal','true');overlay.setAttribute('aria-label','Penalty awarded');
 const dialog=document.createElement('div');dialog.className='game-over-dialog';
 const title=document.createElement('h2');title.textContent='❌ Penalty!';
 dialog.appendChild(title);
 for(const change of changes){
  const message=document.createElement('p');
  message.textContent=change.name+' received '+change.count+' X'+(change.count===1?'':'s')+' (−'+(change.count*5)+' points).';
  dialog.appendChild(message);
 }
 const button=document.createElement('button');button.textContent='Continue';button.onclick=()=>overlay.remove();
 dialog.appendChild(button);overlay.appendChild(dialog);document.body.appendChild(overlay);button.focus();
}
async function act(a){if(moving||!state||state.done)return;moving=true;thinkingActor=null;render();const sid=state.session_id;let polling=false;const poll=async()=>{if(polling||!moving||state?.session_id!==sid)return;polling=true;try{const response=await fetch('/api/session-progress?session_id='+encodeURIComponent(sid),{cache:'no-store'});if(response.ok){const progress=await response.json();if(moving&&state?.session_id===sid&&thinkingActor!==progress.actor){thinkingActor=progress.actor;render()}}}catch(e){console.debug('Thinking progress unavailable',e)}finally{polling=false}};const interval=setInterval(poll,200);void poll();try{const previous=state;const next=await api('/move',{action:a,session_id:sid,expected_turn:state.turn,expected_phase:state.phase});if(next){state=next;pending=null;recordMoveChanges(previous,next);render();showNewPenalties(previous,next)}}finally{clearInterval(interval);moving=false;thinkingActor=null;if(state)render()}}

function selectMark(action){if(moving||!state||state.done)return;if(quickMarkEnabled()&&action[2]&& !window.confirm('Close this row? This cannot be undone.'))return;
 if(quickMarkEnabled()){pending=action;act(action);return}
 pending=pending&&pending[0]===action[0]&&pending[1]===action[1]?null:action;render()}
function clearSelection(){pending=null;render()}
function confirmPhase(){if(pending)act(pending)}
async function backToStep1(){if(moving||!state||state.done||!state.can_undo_step1)return;moving=true;try{const next=await api('/undo-step1',{session_id:state.session_id,expected_turn:state.turn,expected_phase:state.phase});if(next){state=next;pending=null;render()}}finally{moving=false;if(state)render()}}
function skipPhase(){if(!state||state.done||moving)return;act(null)}
function maybeAutoPass(){if(!state||state.done||moving||state.actor!==0||state.legal.length!==0){if(autoPassTimer){clearTimeout(autoPassTimer);autoPassTimer=null}return}const key=state.session_id+':'+state.turn+':'+state.phase;if(autoPassKey===key)return;autoPassKey=key;if(autoPassTimer)clearTimeout(autoPassTimer);autoPassTimer=setTimeout(()=>{autoPassTimer=null;if(!state||state.done||moving||state.actor!==0||state.legal.length!==0)return;const current=state.session_id+':'+state.turn+':'+state.phase;if(current!==key)return;act(null)},650)}
const colors=['red','yellow','green','blue'],names=['Red','Yellow','Green','Blue'];
const playerNames=['You','AI-Sir Rolls-a-Lot','AI-Dicey McDiceface','AI-Count von Qwixx'];
function lockTooltip(color){const entries=(state.action_history||[]).filter(e=>e.action&&e.action[0]===colors.indexOf(color)&&e.action[2]===true);if(!entries.length)return 'Row locked · player unknown';const first=entries[0];const sameRoll=entries.filter(e=>e.turn===first.turn&&e.phase<state.boards.length);const people=[...new Set((sameRoll.length?sameRoll:[first]).map(e=>playerNames[e.actor]||'AI '+e.actor))];return 'Locked by '+people.join(', ')+' · Turn '+first.turn+(first.phase<state.boards.length?' · Step 1':' · Step 2')}
const pipPositions={1:[5],2:[1,9],3:[1,5,9],4:[1,3,7,9],5:[1,3,5,7,9],6:[1,3,4,6,7,9]};
function die(value,color,label,disabled){const locked=state.locks.includes(color);return '<div class="die-wrap"><div class="die '+color+(disabled?' off':'')+(locked?' locked-die':'')+'" aria-label="'+label+' die shows '+value+(locked?', locked':'')+'">'+pipPositions[value].map(p=>'<span class="pip" style="grid-area:'+Math.ceil(p/3)+' / '+((p-1)%3+1)+'"></span>').join('')+'</div><span class="die-label">'+label+'</span></div>'}
function ordinal(n){const mod100=n%100;if(mod100>=11&&mod100<=13)return n+'th';return n+(n%10===1?'st':n%10===2?'nd':n%10===3?'rd':'th')}
function playerRank(i){const score=state.scores[i];const higher=state.scores.filter(s=>s>score).length;const tied=state.scores.filter(s=>s===score).length>1;return {rank:higher+1,tied,label:(tied?'T-':'')+ordinal(higher+1)+' of '+state.scores.length}}
function rankBadge(i,extraClass=''){const r=playerRank(i);return '<span class="rank-badge '+extraClass+(r.rank===1?' leader':'')+'" aria-label="Current position '+r.label+'">'+r.label+'</span>'}
function lockReadyRows(board){return board.rows.map((row,c)=>({c,row})).filter(x=>x.row.length>=5&&!state.locks.includes(colors[x.c]))}
function card(board,i){let b='<div class="player-panel '+(i!==0&&i!==state.roller?'inactive':'active')+'"><h3 class="player-heading"><span>'+playerNames[i]+' — '+state.scores[i]+' points · '+board.penalties+' penalties</span>'+rankBadge(i,i===0?'you':'')+'</h3><section class="scorecard" aria-label="'+(i===0?'Your':'AI '+i)+' Qwixx scorecard">';
 board.rows.forEach((row,c)=>{let nums=c<2?[2,3,4,5,6,7,8,9,10,11,12]:[12,11,10,9,8,7,6,5,4,3,2],last=row.length?nums.indexOf(row[row.length-1]):-1,locked=state.locks.includes(colors[c]),pendingLock=locked&&state.phase<state.boards.length&&(state.white_locks||[]).includes(colors[c]),rowClosed=locked&&!pendingLock,personalLock=locked&&row.includes(nums[10]);
 b+='<div class="score-row '+colors[c]+(rowClosed?' row-locked':'')+'" title="'+(rowClosed?escapeHtml(lockTooltip(colors[c])):'')+'" aria-label="'+names[c]+' row'+(rowClosed?' locked':pendingLock?' lock pending':'')+'">'+nums.map((n,j)=>{const action=i===0&&!state.done?state.legal.find(a=>a[0]===c&&a[1]===n):null;const selected=action&&pending&&pending[0]===c&&pending[1]===n;const classes='number-cell '+(row.includes(n)?'marked':j<last?'skipped':'')+(action?' playable':'')+(selected?' selected':'');const label=names[c]+' '+n+(row.includes(n)?' marked':action?' available to mark':'');return action?'<button type="button" class="'+classes+'" aria-label="'+label+'" onclick="selectMark(['+action.join(',')+'])"><span class="number-label">'+n+'</span></button>':'<div class="'+classes+'" aria-label="'+label+'"><span class="number-label">'+n+'</span></div>'}).join('')+'<div class="lock-cell '+(locked?'locked':row.length>=5?'lock-ready':'')+'" title="'+(!locked&&row.length>=5?'Can lock '+names[c]+' row · '+row.length+' marks':'')+'" aria-label="'+names[c]+' padlock'+(!locked&&row.length>=5?', lock ready':'')+'">'+(locked?'🔒':row.length>=5?'🔓':'♙')+'</div>'+(rowClosed?'<div class="row-lock-overlay" title="'+escapeHtml(lockTooltip(colors[c]))+'" aria-label="'+escapeHtml(lockTooltip(colors[c]))+'"><span>LOCKED</span></div>':'')+'</div>'});
 b+='<div class="score-footer combined"><strong>Marks → points:</strong><div class="score-key">'+Array.from({length:12},(_,j)=>'<span>'+(j+1)+'×<br><b>'+((j+1)*(j+2)/2)+'</b></span>').join('')+'</div><div class="missed-throws"><strong>Missed throws (−5 each):</strong>'+Array.from({length:4},(_,j)=>'<span class="penalty-box '+(j<board.penalties?'used':'')+'">'+(j<board.penalties?'✕':'')+'</span>').join('')+'</div></div>';
 b+='<div class="score-footer score-totals"><strong>Result:</strong>'+board.rows.map((row,c)=>{let k=row.length+(state.locks.includes(colors[c])&&row.includes(c<2?12:2)?1:0);return '<span class="total-box '+colors[c]+'">'+(k*(k+1)/2)+'</span>'}).join('<span>+</span>')+'<span>−</span><span class="total-box penalty">'+(board.penalties*5)+'</span><span>=</span><span class="total-box final">'+state.scores[i]+'</span></div></section></div>';return b}
function miniCard(board,i){
 const leading=state.scores[i]===Math.max(...state.scores),ready=lockReadyRows(board);
 let b='<section class="mini-card'+(state.roller===i?' rolling':'')+(leading?' leading':'')+(moving&&thinkingActor===i?' ai-active':'')+(state.roller!==i&&!(moving&&thinkingActor===i)?' ai-inactive':'')+'" aria-label="'+playerNames[i]+' miniature scorecard"><div class="mini-header"><span>'+playerNames[i]+(moving&&thinkingActor===i?' <strong class="active-ai-badge">● THINKING</strong>':state.roller===i?' <strong class="active-ai-badge">🎲 ROLLING</strong>':'')+'</span><span class="mini-header-meta"><span>'+state.scores[i]+' pts'+(leading?' ★':'')+'</span>'+rankBadge(i)+'</span></div>'+(ready.length?'<div class="lock-threats" aria-label="Rows this player can lock">'+ready.map(x=>'<span class="lock-threat-badge" title="'+playerNames[i]+' has '+x.row.length+' marks in '+names[x.c]+' and can lock the row">⚠ LOCK READY · '+names[x.c].toUpperCase()+'</span>').join('')+'</div>':'');
 board.rows.forEach((row,c)=>{
  const nums=c<2?[2,3,4,5,6,7,8,9,10,11,12]:[12,11,10,9,8,7,6,5,4,3,2];
  const last=row.length?nums.indexOf(row[row.length-1]):-1,locked=state.locks.includes(colors[c]),readyToLock=!locked&&row.length>=5;
  b+='<div class="mini-row '+colors[c]+(locked?' mini-row-locked':'')+(readyToLock?' lock-ready':'')+(recentChanges.some(e=>e.type==='lock'&&e.color===c)?' recent-lock':'')+'" title="'+(locked?escapeHtml(lockTooltip(colors[c])):readyToLock?'Can lock '+names[c]+' row · '+row.length+' marks':'')+'" aria-label="'+names[c]+' row'+(locked?' locked':readyToLock?' lock ready':'')+'">';
  b+=nums.map((n,j)=>'<span class="mini-number'+(row.includes(n)?' marked':(locked||j<last)?' skipped':'')+(recentMark(i,c,n)?' recent-mark':'')+'" aria-label="'+names[c]+' '+n+(row.includes(n)?' marked':(locked||j<last)?' skipped':' available')+'">'+n+'</span>').join('');
  b+='<span class="mini-lock'+(readyToLock?' lock-ready':'')+'" title="'+(readyToLock?'Can lock '+names[c]+' row · '+row.length+' marks':'')+'" aria-label="'+names[c]+(locked?' locked':readyToLock?' lock ready':' unlocked')+'">'+(locked?'🔒':readyToLock?'🔓':'')+'</span>'+(locked?'<span class="mini-locked-label" title="'+escapeHtml(lockTooltip(colors[c]))+'" aria-label="'+escapeHtml(lockTooltip(colors[c]))+'">🔒 LOCKED</span>':'')+'</div>';
 });
 return b+'<div class="mini-footer"><span>Strikes: '+board.penalties+' / 4</span><span>Position: '+playerRank(i).label+'</span></div></section>';
}
function closeGameOver(){document.getElementById('game-over-modal')?.remove()}
function showGameOver(){if(gameOverShown||!state.done)return;gameOverShown=true;const rankings=state.scores.map((score,i)=>({name:playerNames[i],score})).sort((a,b)=>b.score-a.score);const overlay=document.createElement('div');overlay.id='game-over-modal';overlay.className='game-over-overlay';overlay.setAttribute('role','dialog');overlay.setAttribute('aria-modal','true');overlay.setAttribute('aria-label','Game over');overlay.innerHTML='<div class="game-over-dialog"><h2>🏁 Game over!</h2><p><strong>'+rankings[0].name+' wins!</strong></p><p>'+rankings.map(p=>p.name+': '+p.score).join('<br>')+'</p><button onclick="reset()">Play again</button><button onclick="closeGameOver()">View final scorecards</button></div>';document.body.appendChild(overlay)}
function render(){if(!state)return;
 document.documentElement.dataset.reducedMotion=reducedMotion()?'on':'off';
 const roller=playerNames[state.roller]||'AI';const whitePhase=state.phase<state.boards.length;const youRoll=state.roller===0;const noLegal=!state.done&&state.actor===0&&state.legal.length===0;
 const instruction=state.done?'Game over':whitePhase?'White dice: '+state.dice[0]+' + '+state.dice[1]+' = '+(state.dice[0]+state.dice[1]):'Choose one white die plus a matching colored die.';
 const rollerLabel=youRoll?'YOU':roller;
 const turnHeader='<div class="roller-header"><span class="turn-count">TURN '+(state.turn+1)+'</span><span class="roller-name '+(youRoll?'human':'opponent')+'">'+rollerLabel+'</span><span class="roller-caption">'+(youRoll?'are rolling':'is rolling')+'</span></div>';
 const thinkingHtml=moving?'<span class="ai-thinking" role="status" aria-live="polite"><span class="ai-thinking-spinner" aria-hidden="true"></span><span>'+(thinkingActor===null?'AI opponents thinking…':playerNames[thinkingActor]+' thinking…')+'</span></span>':'';
 const diceHtml='<div class="compact-dice-layout"><div class="compact-turn">'+turnHeader+'<p class="turn-subtitle">'+(whitePhase?'Step 1 · White dice total':'Step 2 · White + colored dice')+'</p></div><div class="dice-tray">'+state.dice.map((n,i)=>die(n,i<2?'white':colors[i-2],i<2?'White '+(i+1):names[i-2],i>=2&&(!youRoll||whitePhase||state.locks.includes(colors[i-2]))  )).join('')+'</div></div>';
 document.getElementById('status').innerHTML='';
 const moveHint=noLegal?'No legal mark is available in this phase.':youRoll&&!whitePhase&&state.active_marks===0?'Passing now gives you a strike (−5 points).':youRoll&&whitePhase?'A strike only occurs if you pass both opportunities on your roll.':youRoll?'You already marked this roll, so passing Step 2 has no penalty.':'No strike when you pass on another player’s roll.';
 const selectedText=noLegal?'No legal marks':pending?'Selected: '+names[pending[0]]+' '+pending[1]:'Select a highlighted number';
 const buttons=noLegal?'<div class="auto-pass-status" role="status" aria-live="polite">⚡ AUTO PASS · No legal marks available</div>':'<div class="move-buttons"><button class="confirm" onclick="confirmPhase()" '+(!pending||moving||state.done?'disabled':'')+'>✓ Mark</button><button onclick="clearSelection()" '+(!pending||moving||state.done?'disabled':'')+'>Clear</button><button class="skip" onclick="skipPhase()" '+(moving||state.done?'disabled':'')+'>'+(youRoll&&!whitePhase&&state.active_marks===0?'Pass — take strike':'Pass →')+'</button>'+(state.can_undo_step1?'<button onclick="backToStep1()">← Back to Step 1</button>':'')+'</div>';
 const controls='<div class="dice-move-controls" aria-label="Your move"><div class="move-status"><span class="phase-instruction">'+instruction+'</span><span class="selected-mark" aria-live="polite">'+selectedText+'</span></div>'+buttons+'<div class="move-hint'+(youRoll&&!whitePhase&&state.active_marks===0&&!noLegal?' strike-warning':'')+'">'+moveHint+'</div><div class="dice-shortcuts">Enter: Mark · Esc: Clear · Space: Pass · Ctrl/Cmd+Z: Undo</div></div>';
 const assistance=preference(ASSIST_KEY,'off')==='on'&&state.explanations?.length?state.explanations.filter(x=>Number.isFinite(x.immediate)).sort((a,b)=>b.immediate-a.immediate).slice(0,3):[];
 const choices=state.legal.map(a=>{const c=a[0],n=a[1],details=state.explanations.find(x=>x.move===colors[c]+' '+n),sources=whitePhase?'white '+state.dice[0]+' + white '+state.dice[1]:[0,1].filter(w=>state.dice[w]+state.dice[c+2]===n).map(w=>'white '+(w+1)+' ('+state.dice[w]+') + '+colors[c]+' ('+state.dice[c+2]+')').join(' OR ');return '<li><strong>'+names[c]+' '+n+(a[2]?' 🔒':'')+'</strong> — '+sources+(details?' · +'+details.immediate+' immediate points · skips '+details.skipped:'')+'</li>'}).join('');
 const analysisHtml=state.done?'<section class="move-analysis-panel"><b>Game over. Start a new game.</b></section>':'<section class="move-analysis-panel" aria-label="Move analysis"><div class="move-analysis-actions"><button type="button" onclick="focusLegalMoves()" '+(noLegal?'disabled':'')+'>'+state.legal.length+' legal move'+(state.legal.length===1?'':'s')+' · Show on board</button>'+(noLegal?'<strong>Auto passing this phase</strong>':'')+'</div>'+(assistance.length?'<p class="move-assist">Top immediate-point moves: '+assistance.map(x=>escapeHtml(x.move)+' (+'+x.immediate+')').join(' · ')+'</p>':'')+'<details class="move-analysis"><summary>Move analysis · '+state.legal.length+' available scorecard move'+(state.legal.length===1?'':'s')+'</summary>'+(choices?'<ul>'+choices+'</ul>':'<p>No legal marks are available in this phase.</p>')+'<p>Equivalent dice combinations leading to the same number are shown together.</p></details></section>';
 document.getElementById('boards').innerHTML='<div class="game-grid"><div class="human-column"><div class="dice-status">'+diceHtml+thinkingHtml+controls+'<div class="game-events" role="status" aria-live="polite">'+recentChanges.slice(-4).map(e=>'<span class="game-event '+e.type+'">'+escapeHtml(e.label)+'</span>').join('')+'</div></div>'+card(state.boards[0],0)+analysisHtml+'</div><aside class="opponents-column" aria-label="AI opponents">'+state.boards.slice(1).map((board,j)=>miniCard(board,j+1)).join('')+'</aside></div>';
 const history=document.getElementById('game-history');
 if(history){const actions=state.action_history||[];history.textContent=actions.length?actions.slice(-12).map(e=>'Turn '+e.turn+' · '+(e.phase<state.boards.length?'Step 1':'Step 2')+' · '+(e.source==='human'?'You':'AI '+e.actor)+' · '+(e.action?names[e.action[0]]+' '+e.action[1]:'Passed')+' · whites '+e.white_total).join('\\n'):'No moves recorded yet';}
 if(state.done){showGameOver();if(!document.getElementById('career').hidden)showStats()}
 maybeAutoPass();
}

loadPlayers().then(()=>{if(!authAvailable)return reset();if(maybeShowWelcome())return null;return reset()}).catch(e=>showGameError(e.message));</script><details id="rules-faq"><summary>Qwixx rules & FAQ</summary>
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
        self.model=model;self.difficulty=difficulty;self.lock=threading.Lock();self.game=Game();self.history=[];self.step1_snapshot=None;self.thinking_actor=None
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
        try:
            while not self.game.done and self.game.actor!=0:
                self.thinking_actor=self.game.actor
                self.apply_action(self.ai(self.game),'ai')
        finally:
            self.thinking_actor=None
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
        return {'turn':g.turn,'phase':g.phase,'actor':g.actor,'roller':g.active,'active_marks':g.active_marks,'can_undo_step1':self.can_undo_step1(),'dice':g.dice,'locks':[COLORS[c] for c in sorted(g.locks)],'white_locks':[COLORS[c] for c in sorted(g.white_locks)],'done':g.done,'scores':g.scores(),'boards':[{'rows':b.rows,'penalties':b.penalties} for b in g.boards],'legal':g.legal() if not g.done else [],'difficulty':self.difficulty,'action_history':self.history[-25:],'explanations':self.explanations() if not g.done else []}

def serve(host='127.0.0.1',port=8765,model_path=None):
    session=Session(load(model_path) if model_path else None)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/favicon.svg':
                body=FAVICON_SVG.encode();self.send_response(200);self.send_header('Content-Type','image/svg+xml');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
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
