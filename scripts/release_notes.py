#!/usr/bin/env python3
"""Generate grouped release notes from conventional commit subjects."""
import os
import re
import subprocess
import sys

GROUPS = [
    ("feat", "✨ Features"),
    ("fix", "🐛 Bug fixes"),
    ("perf", "⚡ Performance"),
    ("refactor", "♻️ Refactoring"),
    ("security", "🔒 Security"),
    ("docs", "📚 Documentation"),
    ("test", "✅ Tests"),
    ("build", "📦 Build system"),
    ("ci", "👷 Continuous integration"),
    ("chore", "🧹 Maintenance"),
    ("revert", "↩️ Reverts"),
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
            item = f"- **{scope}:** {description}" if scope else f"- {description}"
            item += f" ({sha})"
            groups.get(kind, groups["other"]).append(item)
            if bang:
                breaking.append(item)
        else:
            groups["other"].append(f"- {subject} ({sha})")
    print(f"# Qwixx AI {tag.removeprefix('v')}\n")
    if previous:
        print(f"Changes since {previous}.\n")
    if breaking:
        print("## 💥 Breaking changes\n" + "\n".join(breaking) + "\n")
    for key, label in GROUPS + [("other", "🔧 Other changes")]:
        if groups[key]:
            print(f"## {label}\n" + "\n".join(groups[key]) + "\n")
    if not lines:
        print("No commits found for this release.")
    version = tag.removeprefix("v")
    print("\n## 🐳 Docker\n")
    print("```bash")
    print(f"docker pull dwightmulcahy/qwixx-ai:{version}")
    print("```\n")
    print("Multi-architecture image: `linux/amd64` and `linux/arm64`.\n")
    repository = os.environ.get("GITHUB_REPOSITORY", "dwightmulcahy/qwixx_ai_next")
    if previous:
        print(f"**Full changelog:** https://github.com/{repository}/compare/{previous}...{tag}")
    else:
        print(f"**Source:** https://github.com/{repository}/tree/{tag}")


if __name__ == "__main__":
    main()
