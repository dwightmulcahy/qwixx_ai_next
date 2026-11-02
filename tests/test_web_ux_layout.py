"""UX regression checks for the responsive scorecards and theme controls."""
import ast
import shutil
import subprocess
from pathlib import Path

import pytest


def html_source():
    source = Path(__file__).resolve().parents[1] / "qwixx_ai" / "web.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    return ast.literal_eval(next(
        stmt.value for stmt in module.body
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "HTML" for t in stmt.targets)
    ))


def test_board_layout_and_accessible_controls():
    html = html_source()
    for expected in (
        "game-grid", "human-column", "opponents-column",
        "aria-label=\"AI opponents\"", "function miniCard(board,i)",
        "mini-number", "mini-lock", "mini-number.skipped",
        "mini-card.rolling", "mini-card.leading",
        "dice-status", "mini-row{border-radius:5px",
        ".number-cell.skipped:after,.mini-number.skipped:after",
        "aria-label=\"Your move\"", "onclick=\"confirmPhase()\"",
        "onclick=\"clearSelection()\"", "onclick=\"skipPhase()\"",
        "onclick=\"backToStep1()\"", "Pass — take strike",
        "state.boards.slice(1).map((board,j)=>miniCard(board,j+1))",
    ):
        assert expected in html, expected
    assert "state.boards.map(card).join('')" not in html


def test_theme_controls_and_invariant_row_colors():
    html = html_source()
    for expected in (
        "theme-select", "Appearance theme", "Auto (system)",
        "function readTheme()", "function setTheme(v)",
        "function applyTheme(v)", "localStorage.setItem(THEME_KEY,v)",
        "prefers-color-scheme: light", "html[data-theme=\"light\"]",
        "--qw-red:#d83a46", "--qw-yellow:#f4d934",
        "--qw-green:#329753", "--qw-blue:#38568e",
    ):
        assert expected in html, expected
    assert "return 'dark'" in html
    assert "themeSelect.value=readTheme()" in html



def test_play_layout_keeps_turn_controls_with_dice_and_analysis_below_scorecard():
    html = html_source()
    assert "['analysis','Move Analysis']" not in html
    assert "panels.play.appendChild(history.closest('details'))" in html
    assert ".dice-status{position:relative" in html
    assert ".ai-thinking{position:absolute" in html
    assert 'class="dice-move-controls"' in html
    assert "diceHtml+thinkingHtml+controls" in html
    assert "card(state.boards[0],0)+analysisHtml" in html
    assert 'class="move-analysis-panel"' in html
    assert "Enter: Mark · Esc: Clear · Space: Pass · Ctrl/Cmd+Z: Undo" in html
    assert "#actions" not in html
    assert "document.getElementById('actions')" not in html


def test_pass_is_always_the_final_move_button():
    html = html_source()
    controls = html.split("const buttons=state.done?", 1)[1].split("const shortcuts=", 1)[0]
    assert controls.index('onclick="backToStep1()"') < controls.index('onclick="skipPhase()"')
    assert controls.rfind('onclick="skipPhase()"') > controls.rfind('onclick="backToStep1()"')


def test_ai_strikes_use_visual_boxes_and_nonblocking_events():
    html = html_source()
    for expected in (
        "mini-penalties",
        "mini-penalty-box",
        "recent-strike",
        "humanChanges=changes.filter(change=>change.player===0)",
        "if(!humanChanges.length)return",
        "❌ '+playerNames[i]+' takes ",
        " · −'+(strikes*5)",
        "{player:i,name:dialogPlayerName(i)",
    ):
        assert expected in html, expected
    assert "Strikes: '+board.penalties+' / 4" not in html


def test_gameplay_preferences_are_in_a_dialog():
    html = html_source()
    for expected in (
        'id="gameplay-modal"', "Gameplay preferences", "⚙ Gameplay",
        "function openGameplayPreferences()", "function closeGameplayPreferences()",
        "function syncGameplayPreferences()", 'id="pref-quick-mark"',
        'id="pref-ai-pace"', 'id="pref-motion"', 'id="pref-move-hints"', 'id="pref-auto-pass"',
        'id="pref-optimal-new-game"',
        "Quick Mark", "AI pace", "Motion", "Move hints", "Auto Pass",
        "Find optimal first roll on New Game",
        "Automatically pass when no legal mark is available.",
        "OPTIMAL_NEW_GAME_KEY='qwixx-optimal-new-game'",
        "preference(OPTIMAL_NEW_GAME_KEY,'off')==='on'",
    ):
        assert expected in html, expected
    assert ".gameplay-overlay[hidden]{display:none!important}" in html


def test_toolbar_order_compact_dice_and_clean_lock_glyphs():
    html = html_source()
    setup = html.split("document.addEventListener('DOMContentLoaded'", 1)[1].split("})();", 1)[0]
    assert setup.index("'#new-game-button'") < setup.index("gameplayButton.textContent='⚙ Gameplay'") < setup.index("themeLabel.textContent='Appearance'")
    for expected in (
        ".gameplay-button{margin-left:0}",
        ".dice-status{position:relative;padding:7px 10px",
        "@media(min-width:761px){.dice-status{height:152px;overflow:hidden}",
        ".compact-dice-layout{display:grid;grid-template-columns:270px minmax(0,1fr)",
        ".dice-status .die{width:46px;height:46px",
        ".dice-move-controls{border-top:1px solid var(--border);margin-top:6px;padding-top:6px",
        ".lock-glyph{position:relative",
        ".lock-glyph.ready:before",
        ".lock-cell{border-radius:8px",
        "class=\"lock-glyph",
    ):
        assert expected in html, expected
    assert "?'🔒':row.length>=5?'🔓':'♙'" not in html


def test_dice_tile_reserves_stable_desktop_height():
    html = html_source()
    for expected in (
        ".dice-status{height:152px;overflow:hidden}",
        ".compact-dice-layout{height:58px}",
        ".dice-move-controls{height:51px",
        ".game-events{height:17px;overflow:hidden;flex-wrap:nowrap}",
        ".compact-turn .roller-header{margin:0 0 2px;gap:5px;flex-wrap:nowrap",
        ".dice-move-controls .move-status{grid-area:status;display:flex;flex-wrap:nowrap",
    ):
        assert expected in html, expected


def test_new_game_dialog_offers_optimal_first_roll():
    html = html_source()
    for expected in (
        'id="new-game-button"', 'onclick="openNewGameDialog()"',
        'id="new-game-modal"', "Start a new game?", "Random first roll",
        "Find optimal first roll", "function openNewGameDialog()",
        "function startNewGame(optimalFirstRoll)",
        "optimal_first_roll:!!optimalFirstRoll",
        'id="finding-roll-modal"', "Finding optimal first roll…",
        "function showFindingOptimalRoll()", "function hideFindingOptimalRoll()",
        "showFindingOptimalRoll();await nextBrowserPaint()",
        "function optimalNewGameEnabled()",
        "if(optimalNewGameEnabled()){void startNewGame(true);return}",
        '<button onclick="openNewGameDialog()">Play again</button>',
    ):
        assert expected in html, expected


def test_mark_renders_before_thinking_indicator():
    html = html_source()
    for expected in (
        "optimisticAction=null", "thinkingVisible=false",
        "function nextBrowserPaint()", "optimisticAction=a===null?null:a",
        "if(optimisticAction){render();await nextBrowserPaint()}",
        "thinkingVisible=true;render()",
        "moving&&thinkingVisible?",
        "optimistic-mark",
    ):
        assert expected in html, expected


def test_dialogs_personalize_the_human_player_name():
    html = html_source()
    for expected in (
        "function humanDialogName()",
        "function dialogPlayerName(i)",
        "function humanDialogPossessive()",
        "verifiedPlayer&&playMode()==='registered'&&verifiedPlayer.name",
        ":'Guest'",
        "dialogPlayerName(i),count:board.penalties-before",
        "escapeHtml(dialogPlayerName(x.i))",
        "escapeHtml(dialogPlayerName(p.actor))",
        "rankings=state.scores.map((score,i)=>({name:dialogPlayerName(i),score}))",
        "winners.length===1?escapeHtml(winners[0].name)+' wins!'",
        "tie for 1st!",
    ):
        assert expected in html, expected
    assert "rankings=state.scores.map((score,i)=>({name:playerNames[i],score}))" not in html
    assert "x.i===0?'You':escapeHtml(playerNames[x.i])" not in html
    assert "p.actor===0?'You':escapeHtml(playerNames[p.actor]" not in html


def test_penalty_pass_and_lock_use_styled_confirmation_dialogs():
    html = html_source()
    for expected in (
        'id="decision-modal"', 'role="alertdialog"', ".decision-overlay[hidden]{display:none!important}",
        "function openDecision(", "function confirmDecision()", "function cancelDecision()",
        "function shouldConfirmPenaltyPass()", "state.legal.length>0",
        "Take a missed-throw penalty?", "Pass anyway (−5)",
        "function confirmLockAction(action)", "Lock the '+names[action[0]]+' row?",
        "function projectedLockScores(action)", "function remainingWhiteCanChange(action)",
        "Projected final scores", "second locked row", "would finish",
    ):
        assert expected in html, expected
    assert "window.confirm('Close this row?" not in html


def test_lock_dialog_lists_same_roll_lock_participants():
    html = html_source()
    for expected in (
        "function lockParticipants(action)",
        "function lockParticipantsHtml(action)",
        "Who can close this row?",
        "will close if you continue",
        "already closed this roll",
        "can also close this roll",
        "No other player can close this row on the current white-dice total.",
        "e.turn===currentTurn",
        "for(let phase=state.phase+1;phase<state.boards.length;phase++)",
    ):
        assert expected in html, expected


def test_penalty_pass_confirmation_only_applies_when_pass_would_cost_points():
    html = html_source()
    assert "state.actor===0&&state.phase===state.boards.length&&state.active_marks===0&&state.legal.length>0" in html
    assert "if(shouldConfirmPenaltyPass()){confirmPenaltyPass();return}act(null)" in html


def test_game_ending_lock_warning_accounts_for_remaining_white_decisions():
    html = html_source()
    for expected in (
        "newDistinct=!state.locks.includes(color)",
        "secondLock=newDistinct&&state.locks.length===1",
        "state.phase>=n-1",
        "remainingWhiteCanChange(action)",
        "Remaining players can still change the scores.",
        "No remaining player has a legal white-dice mark, so these scores are final.",
    ):
        assert expected in html, expected


def test_legal_moves_are_visually_prominent():
    html = html_source()
    assert ".number-cell.playable{cursor:pointer;z-index:2;background:#fff!important" in html
    assert "0 0 18px 5px #22c55e99" in html
    assert "transform:scale(1.04)" in html
    assert "@keyframes available-pulse" in html


def test_back_to_step1_resets_auto_pass_dedupe_and_server_controls_availability():
    html = html_source()
    for expected in (
        "if(autoPassTimer){clearTimeout(autoPassTimer);autoPassTimer=null}autoPassKey='';state=next",
        "state.can_undo_step1?'<button onclick=\"backToStep1()\">← Back to Step 1</button>':''",
    ):
        assert expected in html, expected


def test_auto_pass_is_a_gameplay_preference():
    html = html_source()
    for expected in (
        "AUTO_PASS_KEY='qwixx-auto-pass'",
        'id="pref-auto-pass"',
        "preference(AUTO_PASS_KEY,'on')==='on'",
        "preference(AUTO_PASS_KEY,'on')!=='on'",
        "⚡ AUTO PASS · No legal marks available",
        "No legal marks · Pass required",
        "function maybeAutoPass()",
        "state.actor!==0||state.legal.length!==0",
        "act(null)",
        "maybeAutoPass();",
    ):
        assert expected in html, expected


def test_live_rank_and_lock_ready_indicators():
    html = html_source()
    for expected in (
        "function ordinal(n)", "function playerRank(i)", "function rankBadge(i",
        "function lockReadyRows(board)", "LOCK READY", "lock-threat-badge",
        "mini-row.lock-ready", "mini-lock.lock-ready", "lock-cell.lock-ready",
        "Position: '+playerRank(i).label",
    ):
        assert expected in html, expected
    assert "state.scores.filter(s=>s>score).length" in html
    assert "function lockReadyRows(board){if(state.done)return[]" in html
    assert "readyToLock=!state.done&&!locked&&row.length>=5" in html
    assert "locked?'locked':!state.done&&row.length>=5?'lock-ready'" in html
    assert "!state.done&&!locked&&row.length>=5?' ready'" in html


def test_first_visit_guest_or_registration_prompt():
    html = html_source()
    for expected in (
        'id="welcome-modal"', 'Guest Play', 'Register / Sign in',
        "const PLAY_MODE_KEY='qwixx-play-mode'",
        "sessionStorage.getItem(PLAY_MODE_KEY)",
        "sessionStorage.setItem(PLAY_MODE_KEY,mode)",
        "function maybeShowWelcome()",
        ".welcome-overlay[hidden]{display:none!important}",
        "Registered User — ",
        "function chooseGuestPlay()",
        "function choosePrimaryPlay()",
        "setPlayMode('guest')",
        "setPlayMode('registered')",
        "guest:playMode()==='guest'",
        "showTab('account')",
        "document.getElementById('auth-email')?.focus()",
        "if(maybeShowWelcome())return null",
    ):
        assert expected in html, expected
    # Guest choice is session-only; there must be no persistent guest cookie.
    assert "qwixx_guest_choice" not in html
    assert "Max-Age=31536000" not in html
    assert "rememberGuestChoice" not in html
    # The browser never reads the secure authentication cookie directly.
    assert "qwixx_auth" not in html


def test_live_browser_render_and_theme_behavior(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")
    html = html_source()
    # Evaluate the game script in a minimal DOM, exercising actual JS functions.
    scripts = []
    remaining = html
    while "<script>" in remaining:
        remaining = remaining.split("<script>", 1)[1]
        js, remaining = remaining.split("</script>", 1)
        scripts.append(js)
    game_script = next(js for js in scripts if "function render()" in js)
    runner = r"""
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const code=fs.readFileSync(process.argv[2],'utf8');
const elements={};
function el(id){return elements[id]??=( {innerHTML:'',textContent:'',hidden:true,className:'',focus:()=>{},remove:()=>{delete elements[id]}} )}
const stored={};
const context={
 console,Math,Array,JSON,Promise,setTimeout,clearTimeout,setInterval,clearInterval,
 localStorage:{getItem:k=>stored[k]??null,setItem:(k,v)=>{stored[k]=v}},
 sessionStorage:{getItem:k=>stored['session:'+k]??null,setItem:(k,v)=>{stored['session:'+k]=v},removeItem:k=>{delete stored['session:'+k]}},
 document:{
   documentElement:{dataset:{}},
   body:{appendChild:()=>{}},
   getElementById:el,
   querySelector:()=>null,
   createElement:()=>({id:'',className:'',innerHTML:'',setAttribute:()=>{}})
 },
 window:{matchMedia:()=>({matches:true,addEventListener:()=>{}})},
 fetch:()=>Promise.reject(Error('offline')),
};
vm.createContext(context);
vm.runInContext(code.replace("loadPlayers().then(()=>{if(!authAvailable)return reset();if(maybeShowWelcome())return null;return reset()}).catch(e=>showGameError(e.message));",''),context);
assert.equal(context.document.documentElement.dataset.theme,'dark');
vm.runInContext("setTheme('light')",context);
assert.equal(context.document.documentElement.dataset.theme,'light');
assert.equal(stored['qwixx-theme'],'light');
vm.runInContext("setTheme('auto')",context);
assert.equal(context.document.documentElement.dataset.theme,'light');
vm.runInContext("setTheme('dark')",context);
assert.equal(context.document.documentElement.dataset.theme,'dark');
vm.runInContext("setTheme('invalid')",context);
assert.equal(stored['qwixx-theme'],'dark');
const rows=[[],[],[],[]];
const boards=Array.from({length:4},()=>({rows:rows.map(x=>[...x]),penalties:0}));
boards[0].rows[1]=[2,3,4,5,6];
boards[1].rows[0]=[2,4];
boards[1].rows[1]=[2,3,4,5,6];
boards[2].rows[1]=[2,3,4];
boards[3].penalties=2;
const initial={boards,scores:[10,15,12,8],roller:1,phase:0,turn:3,
 dice:[2,4,3,2,1,5],locks:['red'],done:false,legal:[[0,6,false]],
 explanations:[],active_marks:0,can_undo_step1:false,action_history:[]};
vm.runInContext('state='+JSON.stringify(initial)+'; pending=null;',context);
vm.runInContext('render()',context);
const board=el('boards').innerHTML;
assert.equal((board.match(/class="mini-card/g)||[]).length,3);
assert.match(board,/class="human-column"/);
assert.match(board,/class="dice-status"/);
assert.match(board,/dice-tray/);
assert.match(board,/class="dice-move-controls"/);
assert.match(board,/Enter: Mark · Esc: Clear · Space: Pass · Ctrl\/Cmd\+Z: Undo/);
assert.match(board,/class="move-analysis-panel"/);
assert.ok(board.indexOf('class="scorecard"') < board.indexOf('class="move-analysis-panel"'));
assert.match(board,/lock-cell lock-ready/);
assert.match(board,/lock-glyph ready/);
assert.equal(el("status").innerHTML,"");
assert.match(board,/class="opponents-column"/);
assert.match(board,/AI-Sir Rolls-a-Lot/);
assert.match(board,/3rd of 4/);
assert.match(board,/1st of 4/);
assert.match(board,/2nd of 4/);
assert.match(board,/4th of 4/);
assert.match(board,/LOCK READY · YELLOW/);
assert.match(board,/mini-row yellow lock-ready/);
assert.match(board,/mini-lock lock-ready/);
assert.match(board,/Red 3 skipped/);
assert.match(board,/Red locked/);
assert.match(board,/class="mini-strikes"/);
assert.equal((board.match(/mini-penalty-box used/g)||[]).length>=2,true);
assert.match(board,/mini-card rolling leading/);
assert.match(board,/onclick="confirmPhase\(\)"/);
assert.match(board,/onclick="clearSelection\(\)"/);
assert.match(board,/onclick="skipPhase\(\)"/);
assert.doesNotMatch(el('status').innerHTML,/confirmPhase/);
vm.runInContext('selectMark([0,6,false])',context);
assert.match(el('boards').innerHTML,/Selected: Red 6/);
assert.doesNotMatch(el('boards').innerHTML,/confirmPhase\(\)" disabled/);
vm.runInContext('clearSelection()',context);
assert.match(el('boards').innerHTML,/confirmPhase\(\)" disabled/);
vm.runInContext('state.phase=4;state.roller=0;',context);
vm.runInContext('render()',context);
assert.match(el('boards').innerHTML,/Pass — take strike/);
assert.doesNotMatch(el("boards").innerHTML,/onclick="skipPhase\(\)" disabled/);
vm.runInContext("moving=true;render()",context);
assert.match(el("boards").innerHTML,/onclick="skipPhase\(\)" disabled/);
vm.runInContext("moving=false;render()",context);
assert.doesNotMatch(el("boards").innerHTML,/onclick="skipPhase\(\)" disabled/);
vm.runInContext("state.scores=[10,10,5,0];state.phase=0;state.roller=1;render()",context);
const tied=el("boards").innerHTML;
assert.equal((tied.match(/T-1st of 4/g)||[]).length>=2,true);
assert.match(tied,/3rd of 4/);
assert.match(tied,/4th of 4/);
stored['qwixx-auto-pass']='off';
vm.runInContext("state.actor=0;state.legal=[];state.done=false;moving=false;render()",context);
const manualPass=el("boards").innerHTML;
assert.match(manualPass,/No legal marks · Pass required/);
assert.match(manualPass,/onclick="skipPhase\(\)"/);
assert.doesNotMatch(manualPass,/⚡ AUTO PASS/);
stored['qwixx-auto-pass']='on';
vm.runInContext("render();if(autoPassTimer){clearTimeout(autoPassTimer);autoPassTimer=null}",context);
const automaticPass=el("boards").innerHTML;
assert.match(automaticPass,/⚡ AUTO PASS · No legal marks available/);

// Dialogs use the registered player's display name instead of the generic "You".
vm.runInContext("verifiedPlayer={name:'dWiGhT'};setPlayMode('registered')",context);
assert.equal(vm.runInContext("humanDialogName()",context),"dWiGhT");
assert.equal(vm.runInContext("dialogPlayerName(0)",context),"dWiGhT");

// Passing with a legal mark available on the active player's second step must confirm.
vm.runInContext("state.phase=state.boards.length;state.actor=0;state.roller=0;state.active_marks=0;state.legal=[[0,6,false]];state.scores=[20,25,18,12];moving=false;skipPhase()",context);
assert.equal(el("decision-modal").hidden,false);
assert.equal(el("decision-title").textContent,"Take a missed-throw penalty?");
assert.match(el("decision-body").innerHTML,/dWiGhT/);
assert.match(el("decision-body").innerHTML,/dWiGhT&#39;s score/);
assert.match(el("decision-body").innerHTML,/legal mark/);
assert.match(el("decision-body").innerHTML,/−5 point penalty/);
assert.equal(el("decision-confirm").textContent,"Pass anyway (−5)");
vm.runInContext("cancelDecision()",context);
assert.equal(el("decision-modal").hidden,true);

// A second lock that ends the game with no remaining white actions shows final projected standings.
vm.runInContext("state.phase=state.boards.length;state.actor=0;state.roller=0;state.locks=['red'];state.white_locks=[];state.legal=[[1,12,true]];state.boards[0].rows[1]=[2,3,4,5,6];state.boards[1].rows[2]=[12,11,10,9,8,7,6,5,4];pending=[1,12,true];confirmPhase()",context);
assert.equal(el("decision-modal").hidden,false);
assert.match(el("decision-body").innerHTML,/second locked row/);
assert.match(el("decision-body").innerHTML,/will end the game/);
assert.match(el("decision-body").innerHTML,/Projected final scores/);
assert.match(el("decision-body").innerHTML,/would finish/);
vm.runInContext("cancelDecision()",context);

// During an earlier white-dice decision, remaining players may still change the final scores.
vm.runInContext("state.phase=0;state.actor=0;state.roller=0;state.dice=[6,6,1,1,1,1];state.locks=['red'];state.white_locks=[];state.legal=[[1,12,true]];state.boards[1].rows[1]=[2,3,4,5,6];pending=[1,12,true];confirmPhase()",context);
assert.equal(el("decision-modal").hidden,false);
assert.match(el("decision-body").innerHTML,/scores can still change/);
assert.doesNotMatch(el("decision-body").innerHTML,/Projected final scores/);
vm.runInContext("cancelDecision()",context);

// Same-roll lock dialog identifies everyone who already closed or can still close the row.
vm.runInContext("state.turn=7;state.phase=1;state.actor=0;state.roller=3;state.dice=[6,6,1,1,1,1];state.locks=['yellow'];state.white_locks=['yellow'];state.action_history=[{turn:8,phase:0,actor:3,action:[1,12,true]}];state.boards[0].rows[1]=[2,3,4,5,6];state.boards[1].rows[1]=[2,3,4,5,6];state.boards[2].rows[1]=[2,3,4,5,6];pending=[1,12,true];confirmPhase()",context);
const participants=el("decision-body").innerHTML;
assert.match(participants,/Who can close this row\?/);
assert.match(participants,/dWiGhT/);
assert.doesNotMatch(participants,/>You</);
assert.match(participants,/will close if you continue/);
assert.match(participants,/AI-Count von Qwixx/);
assert.match(participants,/already closed this roll/);
assert.match(participants,/AI-Sir Rolls-a-Lot/);
assert.match(participants,/can also close this roll/);
assert.match(participants,/AI-Dicey McDiceface/);
vm.runInContext("cancelDecision()",context);

// Guest play must not leak the authenticated player's name into dialogs.
vm.runInContext("setPlayMode('guest')",context);
assert.equal(vm.runInContext("humanDialogName()",context),"Guest");
assert.equal(vm.runInContext("dialogPlayerName(0)",context),"Guest");
vm.runInContext("setPlayMode('registered')",context);

vm.runInContext("state.done=true;state.locks=['red','yellow','blue'];state.boards[0].rows[2]=[12,11,10,9,8];state.boards[1].rows[2]=[12,11,10,9,8,7,6,5];state.boards[2].rows[2]=[12,11,10,9,8,7,6,5,4];state.boards[3].rows[2]=[12,11,10,9,8,7];render()",context);
const ended=el("boards").innerHTML;
assert.match(ended,/Game complete/);
assert.match(ended,/Final scores are shown below/);
assert.doesNotMatch(ended,/onclick="confirmPhase\(\)"/);
assert.doesNotMatch(ended,/onclick="skipPhase\(\)"/);
assert.doesNotMatch(ended,/LOCK READY/);
assert.doesNotMatch(ended,/lock-ready/);
assert.doesNotMatch(ended,/lock-glyph ready/);
"""
    script = tmp_path / "ux_check.js"
    game = tmp_path / "game.js"
    script.write_text(runner, encoding="utf-8")
    game.write_text(game_script, encoding="utf-8")
    result = subprocess.run(
        [node, str(script), str(game)], capture_output=True,
        text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_simultaneous_white_lock_remains_clickable_in_scorecard():
    """A global lock created during this white phase must not hide a legal move."""
    html = html_source()
    assert "pendingLock=locked&&state.phase<state.boards.length" in html
    assert "(state.white_locks||[]).includes(colors[c])" in html
    assert "rowClosed=locked&&!pendingLock" in html
    assert "i===0&&!state.done?state.legal.find" in html
    assert "rowClosed?' row-locked'" in html
    assert "rowClosed?" in html and "row-lock-overlay" in html


def test_locked_row_tooltips_identify_locking_players():
    html = html_source()
    assert "function lockTooltip(color)" in html
    assert "e.action[2]===true" in html
    assert "people.join(', ')" in html
    assert 'title="'+ "'+escapeHtml(lockTooltip(colors[c]))+" in html


def test_leaderboard_cards_use_single_metric_and_podium_styling():
    html = html_source()
    leaderboard = html.split("async function showLeaderboard(){", 1)[1].split("function escapeHtml(value)", 1)[0]
    for expected in (
        "leaderboard-card-header", "leaderboard-list", "leaderboard-row",
        "leaderboard-rank", "leaderboard-value", "metric-most_wins",
        "metric-highest_score", "metric-total_points", "metric-average_score",
        "metric-average_place", "metric-win_rate", "metric-most_games",
        "🥇", "🥈", "🥉", "🏆", "🎯", "⭐", "📈", "🏁", "🔥", "🎲",
        "m.meta(p)", "m.format(p)",
    ):
        assert expected in leaderboard or expected in html, expected
    assert "<table" not in leaderboard
    assert "<th>Games</th><th>Wins</th>" not in leaderboard
    assert "<th>'+label+'</th><th>Games</th><th>Wins</th>" not in leaderboard
    assert "Most wins</th>" not in leaderboard
    assert "p.wins+' wins',meta:p=>p.games+' games · '+p.win_rate+'% win rate'" in leaderboard
    assert "p.games+' games',meta:p=>p.wins+' wins · '+p.win_rate+'% win rate'" in leaderboard
    assert "key:'average_place',label:'Best average place'" in leaderboard
    assert "p.average_place+' avg place'" in leaderboard


def test_career_stats_show_average_place_and_recent_place():
    html = html_source()
    stats = html.split("async function showStats(){", 1)[1].split("async function showLeaderboard(){", 1)[0]
    assert "['Average place',data.average_place]" in stats
    assert "<th>Average place</th>" in stats
    assert "<th>Place</th><th>Your score</th>" in stats
    assert "d?d.average_place:'—'" in stats
    assert "fmt(g.place)" in stats


def test_leaderboard_defaults_to_champion_and_separates_difficulties():
    html = html_source()
    assert "select?select.value:'champion'" in html
    assert "['champion','🏆 Champion']" in html
    assert "['advanced','Advanced']" in html
    assert "['intermediate','Intermediate']" in html
    assert "['beginner','Beginner']" in html
    assert "['','All difficulties']" not in html
    assert "results never mix across difficulty levels" in html
    assert "?difficulty='+encodeURIComponent(difficulty)" in html
