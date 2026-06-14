#!/usr/bin/env bash
# Render the statML website (HTML only — no PDF book).
#
# The book is knitr-engine, so Python chunks run via reticulate and need the
# mambaforge Python that has the scientific stack. /opt/homebrew/bin is on PATH
# for rsvg-convert (harmless here; kept for any SVG handling).
#
# NOTE: eval:true chapters reuse frozen results (freeze: true). To force a
# chapter to re-execute, delete its freeze dir first:
#   rm -rf _freeze/chapters/<name>
#
# Usage: ./render.sh   (then commit _book/ + _freeze/ and push to main to deploy)
set -euo pipefail

export PATH="/opt/homebrew/bin:$PATH"
export RETICULATE_PYTHON=/Users/bxc331/mambaforge/bin/python3
export QUARTO_PYTHON=/Users/bxc331/mambaforge/bin/python3

quarto render --to html
echo "Done. Now: git add -A && git commit && git push origin main  (deploys to GitHub Pages)"
