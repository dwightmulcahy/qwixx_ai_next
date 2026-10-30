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
