#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build/main
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
