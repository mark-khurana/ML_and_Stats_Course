#!/bin/bash
# Post-install: dependencies not available on conda-forge.
# Run inside an environment that includes the R stack, e.g.:
#   pixi run -e r    post_install   # R users
#   pixi run -e full post_install   # devs (default `pixi run post_install` targets the default env)
set -euo pipefail

# --- R packages on CRAN but better packaged outside of conda-forge ---
Rscript -e 'install.packages(c("keras3", "dcurves", "EValue", "brms", "rstanarm", "bayesplot", "CMAverse"), repos = "https://cloud.r-project.org")'

# keras3 (R) will use the python TensorFlow backend already provided by the
# `python` / `full` environments via reticulate, so no extra backend install
# is needed when run inside one of those environments.

echo "post_install complete."
