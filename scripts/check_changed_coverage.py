"""Enforce coverage of added or modified executable Python statements."""
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def changed_lines(diff):
    result = {}
    path = None
    line = 0
    for entry in diff.splitlines():
        if entry.startswith("diff --git "):
            path = None
        elif entry.startswith("+++ b/"):
            path = entry[6:]
        elif entry.startswith("@@"):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", entry)
            if match is None:
                raise ValueError("Invalid diff hunk")
            line = int(match[1])
        elif path and entry.startswith("+"):
            result.setdefault(path, set()).add(line)
            line += 1
        elif entry.startswith(" "):
            line += 1
    return result


def measure(root, changed):
    covered = total = 0
    files = {}
    for cls in root.findall(".//class"):
        name = cls.attrib["filename"]
        if not name.startswith("qwixx_ai/"):
            name = "qwixx_ai/" + name
        files.setdefault(name, {}).update(
            (int(line.attrib["number"]), int(line.attrib["hits"]))
            for line in cls.findall("./lines/line")
        )
    for path, lines in changed.items():
        if not path.startswith("qwixx_ai/") or not path.endswith(".py"):
            continue
        if path not in files:
            raise ValueError(f"Missing coverage data for {path}")
        for line in lines & files[path].keys():
            total += 1
            covered += files[path][line] > 0
    return covered, total


def main():
    report, base = sys.argv[1:]
    diff = subprocess.check_output(
        ["git", "diff", "--no-ext-diff", "--no-renames", "--unified=0", base, "HEAD", "--", "qwixx_ai"],
        text=True,
    )
    covered, total = measure(ET.parse(Path(report)).getroot(), changed_lines(diff))
    if not total:
        print("No changed executable Python statements")
        return
    print(f"Changed statement coverage: {covered}/{total} ({covered / total:.2%})")
    if covered * 100 < total * 90:
        raise SystemExit("Changed statement coverage below 90%")


if __name__ == "__main__":
    main()
