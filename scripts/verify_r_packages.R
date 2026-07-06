# verify_r_packages.R - Confirm the key course packages attach (used in CI).
# A failed library() call raises an error and makes Rscript exit non-zero.

pkgs <- c("tidyverse", "rms", "glmnet", "survival", "brms", "ranger", "xgboost", "mice")
for (p in pkgs) {
  library(p, character.only = TRUE)
  cat(p, ": OK\n")
}
