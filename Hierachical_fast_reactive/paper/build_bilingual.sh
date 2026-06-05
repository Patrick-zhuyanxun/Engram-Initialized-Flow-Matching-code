#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/build/en" "$SCRIPT_DIR/build/zh"

cd "$SCRIPT_DIR/src"
tectonic main.tex -o ../build/en
tectonic main_zh.tex -o ../build/zh
