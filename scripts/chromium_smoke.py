"""Launch the real Qwixx server and verify browser rendering with Chromium."""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


def dismiss_first_roller(page):
    dialog = page.locator("#first-roller-modal")
    dialog.wait_for(state="visible", timeout=5000)
    assert dialog.get_by_text("First roller", exact=False).is_visible()
    dialog.get_by_role("button", name="Start game").click()
    dialog.wait_for(state="detached", timeout=5000)


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
                    dismiss_first_roller(page)
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
                    assert preferences.get_by_text("Find optimal first roll on New Game", exact=True).is_visible()
                    auto_pass = preferences.locator("#pref-auto-pass")
                    assert auto_pass.is_checked()
                    auto_pass.uncheck()
                    assert page.evaluate("localStorage.getItem('qwixx-auto-pass')") == "off"
                    auto_pass.check()
                    assert page.evaluate("localStorage.getItem('qwixx-auto-pass')") == "on"
                    optimal_pref = preferences.locator("#pref-optimal-new-game")
                    assert not optimal_pref.is_checked()
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
                    dismiss_first_roller(page)
                    assert page.evaluate(
                        """() => state.legal.some(a => {
                            const order = a[0] < 2
                                ? [2,3,4,5,6,7,8,9,10,11,12]
                                : [12,11,10,9,8,7,6,5,4,3,2];
                            return order.indexOf(a[1]) < 3;
                        })"""
                    )

                    # Gameplay preference can make New Game always search for the optimal opening.
                    gameplay.click()
                    preferences.wait_for(state="visible", timeout=5000)
                    optimal_pref = preferences.locator("#pref-optimal-new-game")
                    optimal_pref.check()
                    assert page.evaluate("localStorage.getItem('qwixx-optimal-new-game')") == "on"
                    preferences.get_by_role("button", name="Close gameplay preferences").click()
                    assert preferences.is_hidden()
                    preferred_old_session = page.evaluate("state.session_id")
                    page.get_by_role("button", name="New game").click()
                    assert new_game.is_hidden()
                    finding_roll.wait_for(state="visible", timeout=5000)
                    assert finding_roll.get_by_text("Finding optimal first roll…", exact=True).is_visible()
                    page.wait_for_function(
                        "(old) => state && state.session_id && state.session_id !== old",
                        arg=preferred_old_session,
                        timeout=10000,
                    )
                    assert finding_roll.is_hidden()
                    dismiss_first_roller(page)

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
                            window.__smokeRafQueue = [];
                            window.requestAnimationFrame = cb => {
                                window.__smokeRafQueue.push(cb);
                                return window.__smokeRafQueue.length;
                            };
                            api = async () => new Promise(() => {});
                            render();
                            confirmPhase();
                        }"""
                    )
                    assert page.locator(".number-cell.optimistic-mark").is_visible()
                    assert page.locator(".ai-thinking").count() == 0
                    page.evaluate(
                        "() => { const queued=[...window.__smokeRafQueue]; window.__smokeRafQueue=[]; queued.forEach(cb=>cb(performance.now())); }"
                    )
                    page.evaluate(
                        "() => { const queued=[...window.__smokeRafQueue]; window.__smokeRafQueue=[]; queued.forEach(cb=>cb(performance.now())); }"
                    )
                    page.wait_for_function("document.querySelector('.ai-thinking') !== null", timeout=3000)
                    assert page.locator(".number-cell.optimistic-mark").is_visible()

                    # Reload clears the intentionally stalled test move.
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("document.querySelector('#boards').innerHTML.length > 100", timeout=30000)
                    assert page.locator("#welcome-modal").is_hidden()
                    dismiss_first_roller(page)

                    # AI strikes are nonblocking: event strip + visual X box, no modal.
                    page.evaluate(
                        """() => {
                            const previous = structuredClone(state);
                            const next = structuredClone(state);
                            next.boards[1].penalties = (previous.boards[1].penalties || 0) + 1;
                            next.done = false;
                            state = next;
                            recordMoveChanges(previous, next);
                            render();
                            showNewPenalties(previous, next);
                        }"""
                    )
                    assert page.locator("#penalty-modal").count() == 0
                    ai_card = page.locator(".mini-card").first
                    assert ai_card.locator(".mini-penalty-box.used").count() >= 1
                    assert ai_card.locator(".mini-penalty-box.recent-strike").count() == 1
                    assert "takes a strike · −5" in page.locator(".game-events").inner_text()

                    # Human strikes still use the blocking penalty dialog.
                    page.evaluate(
                        """() => {
                            const previous = structuredClone(state);
                            const next = structuredClone(state);
                            next.boards[0].penalties = (previous.boards[0].penalties || 0) + 1;
                            next.done = false;
                            state = next;
                            recordMoveChanges(previous, next);
                            render();
                            showNewPenalties(previous, next);
                        }"""
                    )
                    penalty = page.locator("#penalty-modal")
                    penalty.wait_for(state="visible", timeout=5000)
                    assert "Penalty!" in penalty.inner_text()
                    page.evaluate("state.done=true;gameOverShown=false;showGameOver()")
                    assert page.locator("#penalty-modal").count() == 0
                    game_over = page.locator("#game-over-modal")
                    game_over.wait_for(state="visible", timeout=5000)
                    game_over.get_by_role("button", name="View final scorecards").click()
                    assert game_over.count() == 0

                    # Reload restores authoritative game state after the client-only transition test.
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("document.querySelector('#boards').innerHTML.length > 100", timeout=30000)
                    assert page.locator("#welcome-modal").is_hidden()
                    dismiss_first_roller(page)

                    # Dialogs use the signed-in display name instead of generic "You".
                    page.evaluate("() => { verifiedPlayer={name:'dWiGhT'}; setPlayMode('registered'); }")

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
                    assert "dWiGhT" in decision.inner_text()
                    assert "dWiGhT's score" in decision.inner_text()
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
                    assert "who can close this row?" in decision.inner_text().lower()
                    assert "dWiGhT" in decision.inner_text()
                    assert "will close if you continue" in decision.inner_text()
                    assert "second locked row" in decision.inner_text()
                    assert "Projected final scores" in decision.inner_text()
                    decision.get_by_role("button", name="Cancel").click()
                    assert decision.is_hidden()

                    # Game Over names the registered player and handles ties correctly.
                    page.evaluate(
                        """() => {
                            state.done = true;
                            state.scores = [84,78,59,53];
                            gameOverShown = false;
                            showGameOver();
                        }"""
                    )
                    game_over = page.locator("#game-over-modal")
                    game_over.wait_for(state="visible", timeout=5000)
                    assert "dWiGhT wins!" in game_over.inner_text()
                    assert "dWiGhT: 84" in game_over.inner_text()
                    assert "You wins!" not in game_over.inner_text()
                    game_over.get_by_role("button", name="View final scorecards").click()
                    assert game_over.count() == 0

                    page.evaluate(
                        """() => {
                            state.scores = [84,84,59,53];
                            gameOverShown = false;
                            showGameOver();
                        }"""
                    )
                    game_over.wait_for(state="visible", timeout=5000)
                    second_player = page.evaluate("playerNames[1]")
                    assert f"dWiGhT and {second_player} tie for 1st!" in game_over.inner_text()
                    game_over.get_by_role("button", name="View final scorecards").click()

                    # Guest mode intentionally hides the authenticated identity.
                    page.evaluate(
                        """() => {
                            setPlayMode('guest');
                            state.scores = [84,78,59,53];
                            gameOverShown = false;
                            showGameOver();
                        }"""
                    )
                    game_over.wait_for(state="visible", timeout=5000)
                    assert "Guest wins!" in game_over.inner_text()
                    assert "dWiGhT" not in game_over.inner_text()
                    game_over.get_by_role("button", name="View final scorecards").click()
                    # This smoke session started as Guest and has no real auth cookie; restore that
                    # mode before the final reload so startup resumes the existing guest game.
                    page.evaluate("setPlayMode('guest')")

                    # Rules content is immediately visible without a second disclosure click.
                    page.get_by_role("tab", name="Rules & FAQ").click()
                    rules_panel = page.locator("#section-rules")
                    assert rules_panel.get_by_role("heading", name="Rules & FAQ", exact=True).is_visible()
                    assert rules_panel.locator("details#rules-faq").count() == 0
                    assert rules_panel.get_by_text("White dice (everyone)", exact=True).is_visible()

                    report_help = rules_panel.locator("#report")
                    assert report_help.get_by_role("heading", name="How to report a bug or improvement", exact=True).is_visible()
                    report_link = report_help.get_by_role("link", name="Report it here", exact=True)
                    assert report_link.is_visible()
                    assert report_link.get_attribute("href") == "https://forms.gle/aW5nNFJWVfLQaX287"

                    # Shop exposes the two Amazon affiliate products with disclosure.
                    page.get_by_role("tab", name="SHOP", exact=True).click()
                    shop_panel = page.locator("#section-shop")
                    assert shop_panel.get_by_role("heading", name="SHOP", exact=True).is_visible()
                    assert "As an Amazon Associate" in shop_panel.inner_text()
                    game_link = shop_panel.get_by_role("link", name="Shop Qwixx on Amazon", exact=True)
                    scorecards_link = shop_panel.get_by_role("link", name="Shop scorecards on Amazon", exact=True)
                    assert "B00J57138C?tag=brewwatersolv-20" in game_link.get_attribute("href")
                    assert "B099NT5LZS?tag=brewwatersolv-20" in scorecards_link.get_attribute("href")
                    assert "sponsored" in game_link.get_attribute("rel")
                    page.get_by_role("tab", name="Play", exact=True).click()

                    # About includes project context and clear contact paths.
                    page.get_by_role("tab", name="About", exact=True).click()
                    about_panel = page.locator("#section-about")
                    assert about_panel.get_by_role("heading", name="About Qwixx AI", exact=True).is_visible()
                    assert about_panel.get_by_role("heading", name="Get in Touch", exact=True).is_visible()
                    assert about_panel.get_by_role("link", name="GitHub Repository").is_visible()
                    assert about_panel.get_by_role("link", name="Report an Issue").is_visible()
                    assert about_panel.get_by_role("link", name="LinkedIn Profile").is_visible()
                    page.get_by_role("tab", name="Play", exact=True).click()

                    # AI Stats aggregates the three fixed opponents and supports difficulty filtering.
                    page.evaluate(
                        """() => {
                            const previousFetch = fetch;
                            fetch = async (url, options) => {
                                if (String(url).startsWith('/api/ai-stats')) {
                                    return new Response(JSON.stringify({
                                        difficulty:'all',
                                        games:165,
                                        players:[
                                            {name:'AI-Sir Rolls-a-Lot',games:165,wins:58,losses:101,ties:6,win_rate:35.2,average_score:88.4,highest_score:167,lowest_score:28,average_place:2.18,current_win_streak:1,longest_win_streak:6},
                                            {name:'AI-Dicey McDiceface',games:165,wins:49,losses:109,ties:7,win_rate:29.7,average_score:84.2,highest_score:159,lowest_score:24,average_place:2.39,current_win_streak:0,longest_win_streak:5},
                                            {name:'AI-Count von Qwixx',games:165,wins:43,losses:116,ties:6,win_rate:26.1,average_score:81.7,highest_score:151,lowest_score:19,average_place:2.52,current_win_streak:0,longest_win_streak:4}
                                        ]
                                    }), {status:200,headers:{'Content-Type':'application/json'}});
                                }
                                return previousFetch(url, options);
                            };
                            showTab('ai-stats');
                        }"""
                    )
                    ai_stats = page.locator("#section-ai-stats")
                    ai_stats.get_by_role("heading", name="🤖 AI Overall Statistics", exact=True).wait_for(state="visible", timeout=5000)
                    ai_stats.locator(".ai-stat-card").nth(2).wait_for(state="visible", timeout=5000)
                    assert ai_stats.locator(".ai-stat-card").count() == 3
                    assert ai_stats.get_by_text("AI-Sir Rolls-a-Lot", exact=True).is_visible()
                    assert ai_stats.get_by_text("165 tracked games in this view", exact=True).is_visible()
                    assert ai_stats.get_by_text("35.2%", exact=True).is_visible()

                    # Leaderboard cards use one dominant metric, podium styling, and no repeated table columns.
                    page.evaluate(
                        """() => {
                            const originalFetch = fetch;
                            const players = [
                                {name:'dWiGhT',games:136,wins:26,ties:3,highest_score:181,average_score:102.4,average_place:2.14,longest_win_streak:7,total_points:13926,win_rate:19.1},
                                {name:'Roll Master',games:98,wins:21,ties:2,highest_score:175,average_score:99.8,average_place:2.31,longest_win_streak:5,total_points:9780,win_rate:21.4},
                                {name:'Dice Queen',games:84,wins:19,ties:1,highest_score:169,average_score:104.1,average_place:2.48,longest_win_streak:4,total_points:8744,win_rate:22.6}
                            ];
                            fetch = async (url, options) => {
                                if (String(url).startsWith('/api/leaderboard')) {
                                    return new Response(JSON.stringify({
                                        difficulty:'champion',
                                        minimum_games_for_rates:5,
                                        registered_players:3,
                                        rankings:{
                                            most_wins:players,
                                            highest_score:players,
                                            total_points:players,
                                            average_score:players,
                                            average_place:players,
                                            longest_win_streak:players,
                                            win_rate:players,
                                            most_games:players
                                        }
                                    }), {status:200,headers:{'Content-Type':'application/json'}});
                                }
                                return originalFetch(url, options);
                            };
                            showTab('leaderboard');
                        }"""
                    )
                    leaderboard = page.locator("#leaderboard")
                    leaderboard.get_by_text("Most wins", exact=True).wait_for(state="visible", timeout=5000)
                    assert leaderboard.locator("table").count() == 0
                    assert leaderboard.locator(".leaderboard-card").count() == 8
                    wins_card = leaderboard.locator(".leaderboard-card.metric-most_wins")
                    assert "🥇" in wins_card.locator(".leaderboard-row").first.inner_text()
                    assert "26 wins" in wins_card.locator(".leaderboard-row").first.inner_text()
                    assert "136 games · 19.1% win rate" in wins_card.locator(".leaderboard-row").first.inner_text()
                    assert wins_card.locator(".leaderboard-row.first").count() == 1
                    assert wins_card.locator(".leaderboard-row.second").count() == 1
                    assert wins_card.locator(".leaderboard-row.third").count() == 1
                    place_card = leaderboard.locator(".leaderboard-card.metric-average_place")
                    assert place_card.get_by_text("Best average place", exact=True).is_visible()
                    assert "2.14 avg place" in place_card.locator(".leaderboard-row").first.inner_text()
                    assert "136 games · 19.1% win rate" in place_card.locator(".leaderboard-row").first.inner_text()
                    streak_card = leaderboard.locator(".leaderboard-card.metric-longest_win_streak")
                    assert streak_card.get_by_text("Longest winning streak", exact=True).is_visible()
                    assert "7 wins" in streak_card.locator(".leaderboard-row").first.inner_text()

                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("document.querySelector('#boards').innerHTML.length > 100", timeout=30000)
                    assert page.locator("#welcome-modal").is_hidden()
                    dismiss_first_roller(page)
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
