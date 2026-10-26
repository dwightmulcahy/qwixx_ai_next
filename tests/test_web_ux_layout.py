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
 console,Math,Array,JSON,Promise,
 localStorage:{getItem:k=>stored[k]??null,setItem:(k,v)=>{stored[k]=v}},
 document:{documentElement:{dataset:{}},getElementById:el},
 window:{matchMedia:()=>({matches:true,addEventListener:()=>{}})},
 fetch:()=>Promise.reject(Error('offline')),
};
vm.createContext(context);
vm.runInContext(code.replace('loadPlayers().then(reset).catch(e=>showGameError(e.message));',''),context);
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
boards[1].rows[0]=[2,4];
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
assert.equal(el("status").innerHTML,"");
assert.match(board,/class="opponents-column"/);
assert.match(board,/AI-Sir Rolls-a-Lot/);
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
