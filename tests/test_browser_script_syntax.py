"""Prevent Python string escaping from silently breaking inline browser JavaScript."""
import ast
import shutil
import subprocess
from pathlib import Path

import pytest


def test_rendered_html_inline_script_is_valid_javascript(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")
    source = Path(__file__).resolve().parents[1] / "qwixx_ai" / "web.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    assignment = next(
        statement for statement in module.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "HTML" for target in statement.targets)
    )
    html = ast.literal_eval(assignment.value)
    assert '<div id="boards">' in html
    assert "function render()" in html
    start = html.index("<script>") + len("<script>")
    end = html.index("</script>", start)
    script = tmp_path / "game.js"
    script.write_text(html[start:end], encoding="utf-8")
    subprocess.run([node, "--check", str(script)], check=True, capture_output=True, text=True)
