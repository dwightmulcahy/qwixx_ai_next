"""Launch the real Qwixx server and verify browser rendering with Chromium."""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    with tempfile.TemporaryDirectory(prefix="qwixx-smoke-") as directory:
        env = dict(os.environ, QWIXX_DATA=str(Path(directory) / "runs"), QWIXX_CENTRAL_WORKERS="0")
        process = subprocess.Popen(
            [sys.executable, "-m", "qwixx_ai.distributed", "--host", "127.0.0.1", "--port", "18765"],
            env=env,
        )
        try:
            for _ in range(90):
                if process.poll() is not None:
                    raise RuntimeError("Qwixx server exited before becoming healthy")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:18765/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    time.sleep(2)
            else:
                raise RuntimeError("Qwixx server did not become healthy")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto("http://127.0.0.1:18765/", wait_until="domcontentloaded")
                    welcome = page.locator("#welcome-modal")
                    welcome.wait_for(state="visible", timeout=30000)
                    assert page.get_by_role("button", name="Guest Play").is_visible()
                    assert page.get_by_role("button", name="Register / Sign in").is_visible()
                    page.get_by_role("button", name="Guest Play").click()
                    page.locator("#boards").wait_for(timeout=30000)
                    page.wait_for_function("document.querySelector('#boards').innerHTML.length > 100", timeout=30000)
                    assert page.evaluate("sessionStorage.getItem('qwixx-play-mode')") == "guest"
                    dice_tile = page.locator(".dice-status")
                    assert dice_tile.locator(".dice-move-controls").is_visible()
                    assert "Enter: Mark" in dice_tile.inner_text()
                    initial_tile_height = dice_tile.bounding_box()["height"]
                    initial_scorecard_y = page.locator(".human-column .scorecard").bounding_box()["y"]
                    assert initial_tile_height == 152
                    page.evaluate(
                        """() => {
                            state.roller = 3;
                            state.phase = state.boards.length;
                            moving = true;
                            thinkingActor = 3;
                            recentChanges = [
                                {type:'mark', label:'AI-Count von Qwixx marked Green 10 with a long status message'},
                                {type:'strike', label:'AI-Dicey McDiceface received a strike'},
                                {type:'lock', label:'Yellow row locked by another player'}
                            ];
                            render();
                        }"""
                    )
                    assert dice_tile.bounding_box()["height"] == initial_tile_height
                    assert page.locator(".human-column .scorecard").bounding_box()["y"] == initial_scorecard_y
                    page.evaluate("moving=false;thinkingActor=null;recentChanges=[];render()")
                    assert dice_tile.bounding_box()["height"] == initial_tile_height
                    assert page.locator(".human-column .scorecard").bounding_box()["y"] == initial_scorecard_y
                    toolbar_order = page.evaluate(
                        "Array.from(document.querySelector('.game-toolbar').children).map("
                        "el => (el.textContent || '').trim())"
                    )
                    assert toolbar_order.index("New game") < toolbar_order.index("⚙ Gameplay") < toolbar_order.index("AppearanceDarkLightAuto (system)")
                    assert page.locator(".human-column .scorecard").is_visible()
                    assert page.locator(".human-column .move-analysis-panel").is_visible()
                    assert page.evaluate(
                        "document.querySelector('.human-column .scorecard').compareDocumentPosition("
                        "document.querySelector('.human-column .move-analysis-panel')) & Node.DOCUMENT_POSITION_FOLLOWING"
                    )
                    gameplay = page.get_by_role("button", name="⚙ Gameplay")
                    assert gameplay.is_visible()
                    gameplay.click()
                    preferences = page.locator("#gameplay-modal")
                    preferences.wait_for(state="visible", timeout=5000)
                    assert preferences.get_by_text("Quick Mark", exact=True).is_visible()
                    assert preferences.get_by_text("AI pace", exact=True).is_visible()
                    assert preferences.get_by_text("Motion", exact=True).is_visible()
                    assert preferences.get_by_text("Move hints", exact=True).is_visible()
                    assert preferences.get_by_text("Auto Pass", exact=True).is_visible()
                    auto_pass = preferences.locator("#pref-auto-pass")
                    assert auto_pass.is_checked()
                    auto_pass.uncheck()
                    assert page.evaluate("localStorage.getItem('qwixx-auto-pass')") == "off"
                    auto_pass.check()
                    assert page.evaluate("localStorage.getItem('qwixx-auto-pass')") == "on"
                    preferences.get_by_role("button", name="Close gameplay preferences").click()
                    assert preferences.is_hidden()

                    # New Game offers a normal roll or a server-selected optimal opening roll.
                    old_session = page.evaluate("state.session_id")
                    page.evaluate(
                        """() => {
                            const originalApi = api;
                            api = async (...args) => {
                                if (args[0] === '/reset' && args[1]?.optimal_first_roll) {
                                    await new Promise(resolve => setTimeout(resolve, 350));
                                }
                                return originalApi(...args);
                            };
                        }"""
                    )
                    page.get_by_role("button", name="New game").click()
                    new_game = page.locator("#new-game-modal")
                    new_game.wait_for(state="visible", timeout=5000)
                    assert new_game.get_by_role("button", name="Random first roll").is_visible()
                    assert new_game.get_by_role("button", name="Find optimal first roll").is_visible()
                    new_game.get_by_role("button", name="Find optimal first roll").click()
                    finding_roll = page.locator("#finding-roll-modal")
                    finding_roll.wait_for(state="visible", timeout=5000)
                    assert finding_roll.get_by_text("Finding optimal first roll…", exact=True).is_visible()
                    assert "leftmost three numbers" in finding_roll.inner_text()
                    page.wait_for_function(
                        "(old) => state && state.session_id && state.session_id !== old",
                        arg=old_session,
                        timeout=10000,
                    )
                    assert new_game.is_hidden()
                    assert finding_roll.is_hidden()
                    assert page.evaluate(
                        """() => state.legal.some(a => {
                            const order = a[0] < 2
                                ? [2,3,4,5,6,7,8,9,10,11,12]
                                : [12,11,10,9,8,7,6,5,4,3,2];
                            return order.indexOf(a[1]) < 3;
                        })"""
                    )

                    # The human mark paints before the AI thinking indicator appears.
                    page.evaluate(
                        """() => {
                            state.actor = 0;
                            state.roller = 0;
                            state.phase = 0;
                            state.done = false;
                            state.legal = [[0,2,false]];
                            state.boards[0].rows[0] = [];
                            pending = [0,2,false];
                            moving = false;
                            thinkingActor = null;
                            thinkingVisible = false;
                            optimisticAction = null;
                            api = async () => new Promise(() => {});
                            render();
                            confirmPhase();
                        }"""
                    )
                    assert page.locator(".number-cell.optimistic-mark").is_visible()
                    assert page.locator(".ai-thinking").count() == 0
                    page.wait_for_function("document.querySelector('.ai-thinking') !== null", timeout=3000)
                    assert page.locator(".number-cell.optimistic-mark").is_visible()

                    # Reload clears the intentionally stalled test move.
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("document.querySelector('#boards').innerHTML.length > 100", timeout=30000)
                    assert page.locator("#welcome-modal").is_hidden()

                    # Passing despite a legal mark and taking a strike uses the styled confirmation dialog.
                    page.evaluate(
                        """() => {
                            state.actor = 0;
                            state.roller = 0;
                            state.phase = state.boards.length;
                            state.active_marks = 0;
                            state.legal = [[0, 6, false]];
                            state.scores = [20, 25, 18, 12];
                            confirmPenaltyPass();
                        }"""
                    )
                    decision = page.locator("#decision-modal")
                    decision.wait_for(state="visible", timeout=5000)
                    assert decision.get_by_text("Take a missed-throw penalty?", exact=True).is_visible()
                    assert "−5 point penalty" in decision.inner_text()
                    assert decision.get_by_role("button", name="Pass anyway (−5)").is_visible()
                    decision.get_by_role("button", name="Cancel").click()
                    assert decision.is_hidden()

                    # A game-ending second lock shows projected standings in the same styled dialog.
                    page.evaluate(
                        """() => {
                            state.actor = 0;
                            state.roller = 0;
                            state.phase = state.boards.length;
                            state.locks = ['red'];
                            state.white_locks = [];
                            state.legal = [[1, 12, true]];
                            state.boards[0].rows[1] = [2,3,4,5,6];
                            state.boards[1].rows[2] = [12,11,10,9,8,7,6,5,4];
                            confirmLockAction([1,12,true]);
                        }"""
                    )
                    decision.wait_for(state="visible", timeout=5000)
                    assert decision.get_by_text("Lock the Yellow row?", exact=True).is_visible()
                    assert "Who can close this row?" in decision.inner_text()
                    assert "will close if you continue" in decision.inner_text()
                    assert "second locked row" in decision.inner_text()
                    assert "Projected final scores" in decision.inner_text()
                    decision.get_by_role("button", name="Cancel").click()
                    assert decision.is_hidden()

                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("document.querySelector('#boards').innerHTML.length > 100", timeout=30000)
                    assert page.locator("#welcome-modal").is_hidden()
                    assert page.locator("body").inner_text().strip()
                    assert not errors, errors
                finally:
                    browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
