# verify_r_packages.R - Confirm key course packages attach (used in CI).
# A failed library() call raises an error and makes Rscript exit non-zero.
#
# Usage:
#   Rscript scripts/verify_r_packages.R              # all packages (Vanilla R)
#   Rscript scripts/verify_r_packages.R --conda-only  # conda-forge packages only (Pixi R)

args <- commandArgs(trailingOnly = TRUE)
conda_only <- "--conda-only" %in% args

conda_pkgs <- c(
  "tidyverse", "rms", "glmnet", "tidymodels", "survminer", "tidycmprsk",
  "mlbench", "kernlab", "rpart.plot", "PRROC", "Rtsne", "uwot", "mclust",
  "MatchIt", "cobalt", "WeightIt", "meta", "metafor", "netmeta",
  "marginaleffects", "ranger", "xgboost", "shapviz", "pdp", "pmsampsize",
  "mice"
)

cran_pkgs <- c(
  "brms", "rstanarm", "bayesplot", "keras3", "dcurves", "EValue",
  "vip", "dagitty", "regmedint", "gtsummary", "CMAverse"
)

pkgs <- if (conda_only) conda_pkgs else c(conda_pkgs, cran_pkgs)

for (p in pkgs) {
  library(p, character.only = TRUE)
  cat(p, ": OK\n")
}
