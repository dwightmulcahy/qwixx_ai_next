import xml.etree.ElementTree as ET

import pytest

from scripts.check_changed_coverage import changed_lines, measure


def test_changed_lines_and_coverage():
    diff = """diff --git a/qwixx_ai/a.py b/qwixx_ai/a.py
--- a/qwixx_ai/a.py
+++ b/qwixx_ai/a.py
@@ -2,1 +2,3 @@
-old
+new
+# comment
+other
"""
    root = ET.fromstring('<coverage><class filename="a.py"><lines><line number="2" hits="1"/><line number="4" hits="0"/></lines></class></coverage>')
    assert changed_lines(diff) == {"qwixx_ai/a.py": {2, 3, 4}}
    assert measure(root, changed_lines(diff)) == (1, 2)
    assert measure(root, {"tests/test_a.py": {1}}) == (0, 0)
    with pytest.raises(ValueError, match="Missing coverage"):
        measure(root, {"qwixx_ai/new.py": {1}})
