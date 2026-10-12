#!/usr/bin/env python3
"""Generate grouped release notes from conventional commit subjects."""
import re
import subprocess
import sys

GROUPS = [
    ("feat", "Features"),
    ("fix", "Bug Fixes"),
    ("perf", "Performance"),
    ("refactor", "Refactoring"),
    ("docs", "Documentation"),
    ("test", "Tests"),
    ("build", "Build"),
    ("ci", "CI"),
    ("chore", "Maintenance"),
    ("revert", "Reverts"),
]
PATTERN = re.compile(r"^(\w+)(?:\(([^)]+)\))?(!)?:\s+(.+)$")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    tag = sys.argv[1]
    tags = git("tag", "--merged", tag, "--sort=-version:refname").splitlines()
    previous = next((t for t in tags if t != tag and re.fullmatch(r"v\d+\.\d+\.\d+(?:-[\w.-]+)?", t)), None)
    commit_range = f"{previous}..{tag}" if previous else tag
    lines = git("log", "--format=%s%x09%h", commit_range).splitlines()
    groups = {key: [] for key, _ in GROUPS}
    groups["other"] = []
    breaking = []
    for line in lines:
        if "\t" not in line:
            continue
        subject, sha = line.rsplit("\t", 1)
        match = PATTERN.match(subject)
        if match:
            kind, scope, bang, description = match.groups()
            item = f"- {description}" + (f" ({scope})" if scope else "") + f" (`{sha}`)"
            groups.get(kind, groups["other"]).append(item)
            if bang:
                breaking.append(item)
        else:
            groups["other"].append(f"- {subject} (`{sha}`)")
    print(f"## {tag}\n")
    if previous:
        print(f"Changes since {previous}.\n")
    if breaking:
        print("### Breaking Changes\n" + "\n".join(breaking) + "\n")
    for key, label in GROUPS + [("other", "Other Changes")]:
        if groups[key]:
            print(f"### {label}\n" + "\n".join(groups[key]) + "\n")
    if not lines:
        print("No commits found for this release.")


if __name__ == "__main__":
    main()
