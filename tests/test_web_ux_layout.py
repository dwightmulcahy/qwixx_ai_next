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


def test_gameplay_preferences_are_in_a_dialog():
    html = html_source()
    for expected in (
        'id="gameplay-modal"', "Gameplay preferences", "⚙ Gameplay",
        "function openGameplayPreferences()", "function closeGameplayPreferences()",
        "function syncGameplayPreferences()", 'id="pref-quick-mark"',
        'id="pref-ai-pace"', 'id="pref-motion"', 'id="pref-move-hints"',
        "Quick Mark", "AI pace", "Motion", "Move hints",
    ):
        assert expected in html, expected
    assert ".gameplay-overlay[hidden]{display:none!important}" in html


def test_toolbar_order_compact_dice_and_clean_lock_glyphs():
    html = html_source()
    setup = html.split("document.addEventListener('DOMContentLoaded'", 1)[1].split("})();", 1)[0]
    assert setup.index("button[onclick=\"reset()\"]") < setup.index("gameplayButton.textContent='⚙ Gameplay'") < setup.index("themeLabel.textContent='Appearance'")
    for expected in (
        ".gameplay-button{margin-left:0}",
        ".dice-status{position:relative;padding:7px 10px",
        ".dice-status .die{width:46px;height:46px",
        ".dice-move-controls{border-top:1px solid var(--border);margin-top:6px;padding-top:6px",
        ".lock-glyph{position:relative",
        ".lock-glyph.ready:before",
        ".lock-cell{border-radius:8px",
        "class=\"lock-glyph",
    ):
        assert expected in html, expected
    assert "?'🔒':row.length>=5?'🔓':'♙'" not in html


def test_legal_moves_are_visually_prominent():
    html = html_source()
    assert ".number-cell.playable{cursor:pointer;z-index:2;background:#fff!important" in html
    assert "0 0 18px 5px #22c55e99" in html
    assert "transform:scale(1.04)" in html
    assert "@keyframes available-pulse" in html


def test_auto_passes_when_human_has_no_legal_mark():
    html = html_source()
    for expected in (
        "const noLegal=!state.done&&state.actor===0&&state.legal.length===0",
        "⚡ AUTO PASS · No legal marks available",
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
    assert "row.length>=5&&!state.locks.includes" in html


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
function el(id){return elements[id]??=( {innerHTML:'',textContent:'',hidden:true} )}
const stored={};
const context={
 console,Math,Array,JSON,Promise,setTimeout,clearTimeout,setInterval,clearInterval,
 localStorage:{getItem:k=>stored[k]??null,setItem:(k,v)=>{stored[k]=v}},
 sessionStorage:{getItem:k=>stored['session:'+k]??null,setItem:(k,v)=>{stored['session:'+k]=v},removeItem:k=>{delete stored['session:'+k]}},
 document:{documentElement:{dataset:{}},getElementById:el,querySelector:()=>null},
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
assert.match(board,/Strikes: 2 \/ 4/);
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
vm.runInContext("state.done=true;render()",context);
const ended=el("boards").innerHTML;
assert.match(ended,/Game complete/);
assert.match(ended,/Final scores are shown below/);
assert.doesNotMatch(ended,/onclick="confirmPhase\(\)"/);
assert.doesNotMatch(ended,/onclick="skipPhase\(\)"/);
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
