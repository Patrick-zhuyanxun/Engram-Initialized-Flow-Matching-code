#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
search_dir="$script_dir"
repo_root=""

while [[ "$search_dir" != "/" ]]; do
  if [[ -f "$search_dir/docs/training.md" && -f "$search_dir/scripts/generate_training_presentation.py" ]]; then
    repo_root="$search_dir"
    break
  fi
  search_dir="$(dirname -- "$search_dir")"
done

if [[ -z "$repo_root" ]]; then
  echo "[hfrvla-training-docs-hook] skipped: repo root not found"
  exit 0
fi

source_md="$repo_root/docs/training.md"
generator="$repo_root/scripts/generate_training_presentation.py"
open_slide_root="$repo_root/docs/presentations/hfrvla-training-open-slide"
open_slide_source="$open_slide_root/slides/hfrvla-training/index.tsx"
open_slide_standalone="$open_slide_root/standalone/entry.tsx"
open_slide_standalone_html="$open_slide_root/standalone/index.html"
open_slide_config="$open_slide_root/open-slide.config.ts"
open_slide_package="$open_slide_root/package.json"
open_slide_lock="$open_slide_root/package-lock.json"
output_html="$repo_root/docs/training_presentation.html"
python_bin="${PYTHON_BIN:-python3}"

if [[ ! -f "$output_html" \
  || "$source_md" -nt "$output_html" \
  || "$generator" -nt "$output_html" \
  || "$open_slide_source" -nt "$output_html" \
  || "$open_slide_standalone" -nt "$output_html" \
  || "$open_slide_standalone_html" -nt "$output_html" \
  || "$open_slide_config" -nt "$output_html" \
  || "$open_slide_package" -nt "$output_html" \
  || "$open_slide_lock" -nt "$output_html" ]]; then
  "$python_bin" "$generator"
  exit 0
fi

"$python_bin" "$generator" --check >/dev/null || "$python_bin" "$generator"
