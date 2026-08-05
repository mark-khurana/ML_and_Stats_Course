#!/bin/bash
# Post-install: dependencies not available on conda-forge.
# Run inside an environment that includes the R stack, e.g.:
#   pixi run -e r    post_install   # R users
#   pixi run -e full post_install   # devs (default `pixi run post_install` targets the default env)
set -euo pipefail

# --- R packages on CRAN but better packaged outside of conda-forge ---
Rscript -e 'install.packages(c("keras3", "dcurves", "EValue", "brms", "rstanarm", "bayesplot", "remotes", "dagitty", "regmedint", "gtsummary"), repos = "https://cloud.r-project.org")'

# --- CMAverse is GitHub-only (not on CRAN) ---
Rscript -e 'if (!requireNamespace("CMAverse", quietly = TRUE)) remotes::install_github("bs1125/CMAverse")'

# pandoc section bibliographies
quarto install extension pandoc-ext/section-bibliographies

echo "post_install complete."
