# install_packages.R — Install all R packages needed for this course.
#
# Open RStudio and run:
#   source("R/install_packages.R")
#
# This may take 10-15 minutes the first time (brms/rstanarm compile Stan).

pkgs <- c(
  # Core tidyverse ecosystem
  "tidyverse",       # dplyr, ggplot2, tidyr, readr, purrr, ...
  "broom",           # tidy model output
  "patchwork",       # combine ggplot panels
  "here",            # project-relative file paths

  # Regression & modeling frameworks
  "rms",             # Regression Modeling Strategies (Harrell)
  "glmnet",          # Lasso / ridge / elastic net
  "tidymodels",      # Unified ML framework
  "caret",           # Classification And REgression Training
  "MASS",            # Modern Applied Statistics (veninger/Ripley)

  # Survival analysis
  "survival",        # Cox models, Kaplan-Meier
  "survminer",       # Survival curve visualization
  "tidycmprsk",      # Competing risks
  "survey",          # Complex survey designs

  # Mixed models & missing data
  "lme4",            # Linear mixed-effects models
  "lmerTest",        # p-values for lme4
  "mice",            # Multiple imputation

  # Trees, ensembles & kernels
  "ranger",          # Fast random forests
  "rpart",           # Recursive partitioning
  "rpart.plot",      # Plotting rpart trees
  "xgboost",         # Gradient boosted trees
  "kernlab",         # Kernel methods (SVM etc.)
  "mlbench",         # ML benchmark datasets

  # Clustering
  "cluster",         # k-medoids (PAM), hierarchical
  "mclust",          # Model-based clustering
  "dbscan",          # Density-based clustering

  # Dimensionality reduction
  "uwot",            # UMAP
  "Rtsne",           # t-SNE

  # Model evaluation & explainability
  "pROC",            # ROC curves / AUC
  "PRROC",           # Precision-recall curves
  "dcurves",         # Decision curve analysis
  "vip",             # Variable importance
  "pdp",             # Partial dependence plots
  "shapviz",         # SHAP values visualization
  "pmsampsize",      # Prediction model sample size

  # Bayesian
  "brms",            # Bayesian regression via Stan
  "rstanarm",        # Pre-compiled Bayesian models
  "bayesplot",       # Bayesian diagnostic plots

  # Meta-analysis
  "metafor",         # Meta-analysis
  "meta",            # Meta-analysis (alternative interface)
  "netmeta",         # Network meta-analysis

  # Causal inference & mediation
  "WeightIt",        # Propensity score weighting
  "MatchIt",         # Propensity score matching
  "cobalt",          # Covariate balance tables/plots
  "marginaleffects", # Marginal effects & contrasts
  "dagitty",         # DAG analysis
  "EValue",          # E-values for unmeasured confounding
  "CMAverse",        # Causal mediation analysis

  # Deep learning
  "keras3",          # Keras interface to TensorFlow

  # Utilities
  "remotes"          # Install packages from GitHub
)

installed <- rownames(installed.packages())
to_install <- pkgs[!pkgs %in% installed]

if (length(to_install) == 0) {
  message("All ", length(pkgs), " packages are already installed.")
} else {
  message("Installing ", length(to_install), " packages: ",
          paste(to_install, collapse = ", "))
  install.packages(to_install, repos = "https://cloud.r-project.org")
}
