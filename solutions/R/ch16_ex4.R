# Chapter 16, Exercise 4: Full Pipeline - PCA + Clustering + UMAP Visualisation
# 800 patients, 30 clinical variables, 4 underlying subtypes

library(tidyverse)
library(cluster)
library(uwot)

# ---- Simulate data ----
set.seed(42)
n <- 800
p <- 30

# 4 subtypes with different prevalences
subtype_probs <- c(0.30, 0.30, 0.25, 0.15)
true_subtype <- sample(1:4, n, replace = TRUE, prob = subtype_probs)

cat("True subtype distribution:\n")
print(table(true_subtype))

# Generate base data
X <- matrix(rnorm(n * p), ncol = p)

# Add subtype-specific signals in different feature subsets
for (s in 1:4) {
  feature_start <- (s - 1) * 6 + 1
  feature_end <- min(s * 6, p)
  X[true_subtype == s, feature_start:feature_end] <-
    X[true_subtype == s, feature_start:feature_end] + 2.5
}

# ---- (a) Standardise and PCA ----
cat("\n=== Part (a): PCA ===\n")

X_scaled <- scale(X)
pca_result <- prcomp(X_scaled)

var_prop <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cum_var <- cumsum(var_prop)

# Scree plot
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
barplot(var_prop[1:15], names.arg = 1:15, col = "steelblue",
        main = "Scree Plot (first 15 PCs)",
        xlab = "Component", ylab = "Proportion of Variance")
abline(h = 1/p, col = "firebrick", lty = 2)

plot(1:15, cum_var[1:15], type = "b", pch = 16, col = "steelblue",
     main = "Cumulative Variance", xlab = "Components",
     ylab = "Cumulative Proportion", ylim = c(0, 1))
abline(h = 0.80, col = "firebrick", lty = 2)

n_80 <- which(cum_var >= 0.80)[1]
cat("PCs needed for 80% variance:", n_80, "\n")

# Use first 10 PCs for clustering
n_pcs <- 10
pca_scores <- pca_result$x[, 1:n_pcs]

# ---- (b) K-means on PCA scores ----
cat("\n=== Part (b): K-means on PCA Scores ===\n")

sil_scores <- numeric(5)
for (k in 2:5) {
  km <- kmeans(pca_scores, centers = k, nstart = 50)
  sil_scores[k] <- mean(silhouette(km$cluster, dist(pca_scores))[, 3])
  cat(sprintf("k = %d: Silhouette = %.3f\n", k, sil_scores[k]))
}

best_k <- which.max(sil_scores[2:5]) + 1
cat("Best k by silhouette:", best_k, "\n")

# Fit final clustering
km_final <- kmeans(pca_scores, centers = best_k, nstart = 50)

# ---- (c) UMAP visualisation ----
cat("\n=== Part (c): UMAP Visualisation ===\n")

set.seed(42)
umap_result <- umap(pca_scores, n_neighbors = 15, min_dist = 0.1,
                     verbose = FALSE)

plot_df <- tibble(
  UMAP1 = umap_result[, 1],
  UMAP2 = umap_result[, 2],
  Cluster = factor(km_final$cluster),
  True_subtype = factor(true_subtype)
)

cols4 <- c("steelblue", "firebrick", "forestgreen", "goldenrod")

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Discovered clusters
plot(umap_result, col = cols4[km_final$cluster], pch = 16, cex = 0.6,
     main = "K-means Clusters on UMAP", xlab = "UMAP 1", ylab = "UMAP 2")
legend("topright", paste("Cluster", 1:best_k), col = cols4[1:best_k],
       pch = 16, cex = 0.7)

# True subtypes
plot(umap_result, col = cols4[true_subtype], pch = 16, cex = 0.6,
     main = "True Subtypes on UMAP", xlab = "UMAP 1", ylab = "UMAP 2")
legend("topright", paste("Subtype", 1:4), col = cols4, pch = 16, cex = 0.7)

# ---- (d) Cluster profiles ----
cat("\n=== Part (d): Cluster Profiles ===\n\n")

# Name variables for interpretability
var_names <- paste0("V", 1:p)
dat_df <- as.data.frame(X)
colnames(dat_df) <- var_names
dat_df$cluster <- km_final$cluster

# Compute mean of each variable by cluster
profiles <- dat_df %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean), n = n(), .groups = "drop")

cat("Cluster sizes:\n")
print(table(km_final$cluster))

cat("\nCluster means for key variables (first 24, showing subtype signals):\n\n")

# Show means for the signal variables
signal_vars <- paste0("V", 1:24)
profile_table <- profiles %>%
  select(cluster, n, all_of(signal_vars))

# Print in a compact format
for (cl in 1:best_k) {
  prof <- profile_table %>% filter(cluster == cl)
  cat(sprintf("Cluster %d (n = %d):\n", cl, prof$n))
  for (s in 1:4) {
    start <- (s - 1) * 6 + 1
    end <- s * 6
    vars <- paste0("V", start:end)
    means <- round(unlist(prof[vars]), 2)
    cat(sprintf("  Subtype %d signal vars (V%d-V%d): mean = %.2f\n",
                s, start, end, mean(means)))
  }
  cat("\n")
}

# ---- (e) Discussion ----
cat("=== Part (e): Validation Discussion ===\n\n")

cat("To validate these clusters in a real clinical study:\n\n")

cat("1. EXTERNAL OUTCOMES: Compare clusters on outcomes NOT used in\n")
cat("   clustering (mortality, length of stay, treatment response).\n")
cat("   If clusters predict outcomes they were not trained on, the\n")
cat("   structure is likely clinically meaningful.\n\n")

cat("2. STABILITY ANALYSIS: Resample the data (bootstrap) and re-run\n")
cat("   clustering. If the same patients consistently end up in the\n")
cat("   same clusters, the results are robust. If clusters change\n")
cat("   substantially across resamples, they may be artefacts.\n\n")

cat("3. INDEPENDENT REPLICATION: Apply the same pipeline to an\n")
cat("   independent dataset (different hospital, different time period).\n")
cat("   If similar clusters emerge, the findings are generalisable.\n\n")

cat("4. CLINICAL EXPERT REVIEW: Present cluster profiles to clinicians.\n")
cat("   Do the clusters correspond to recognisable patient types?\n")
cat("   Do they suggest different management strategies?\n\n")

cat("5. MULTIPLE ALGORITHMS: Compare results from K-means, hierarchical\n")
cat("   clustering, and DBSCAN. Agreement across methods strengthens\n")
cat("   confidence. Disagreement suggests fragile structure.\n\n")

cat("6. AVOID CIRCULAR REASONING: Do NOT validate clusters by showing\n")
cat("   they differ on the same variables used to create them.\n")
cat("   This is guaranteed by construction and proves nothing.\n")
