#!/usr/bin/env sh
# Build with the nearest reachable Git tag; leave tagging to the maintainer.
set -eu
if command -v git >/dev/null 2>&1; then
    version=$(git describe --tags --abbrev=0 2>/dev/null || true)
else
    version=''
fi
QWIXX_GIT_VERSION="${QWIXX_GIT_VERSION:-${version:-dev}}" docker compose up -d --build "$@"
