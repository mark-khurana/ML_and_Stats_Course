# Chapter 15, Exercise 4: Full Workflow on Simulated Genomic Data
# 1000 patients, 500 gene expression features, 4 cancer subtypes

library(tidyverse)
library(Rtsne)
library(uwot)

# ---- Simulate data ----
set.seed(42)
n <- 1000
p <- 500

# 4 cancer subtypes with different prevalences (one rare at 5%)
subtype_probs <- c(0.35, 0.30, 0.30, 0.05)
subtype <- sample(1:4, n, replace = TRUE, prob = subtype_probs)

cat("Subtype distribution:\n")
print(table(subtype))

# Generate base gene expression
X <- matrix(rnorm(n * p), ncol = p)

# Add subtype-specific signals in different gene subsets
# Subtype 1: upregulated in genes 1-30
X[subtype == 1, 1:30] <- X[subtype == 1, 1:30] + 2.0

# Subtype 2: upregulated in genes 31-60, downregulated in 61-80
X[subtype == 2, 31:60] <- X[subtype == 2, 31:60] + 2.5
X[subtype == 2, 61:80] <- X[subtype == 2, 61:80] - 1.5

# Subtype 3: upregulated in genes 81-120
X[subtype == 3, 81:120] <- X[subtype == 3, 81:120] + 2.0

# Subtype 4 (rare): strong signal in genes 121-160
X[subtype == 4, 121:160] <- X[subtype == 4, 121:160] + 3.5

# ---- (a) Scale and PCA ----
cat("\n=== Part (a): PCA ===\n")
X_scaled <- scale(X)
pca_result <- prcomp(X_scaled)

var_prop <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cum_var <- cumsum(var_prop)
n_80 <- which(cum_var >= 0.80)[1]

cat("PCs needed for 80% variance:", n_80, "\n")
cat("First 10 cumulative variance:", round(cum_var[1:10], 3), "\n")

# Scree plot (first 30 components)
par(mfrow = c(1, 1))
barplot(var_prop[1:30], names.arg = 1:30, col = "steelblue",
        main = "Scree Plot (first 30 PCs)",
        xlab = "Component", ylab = "Proportion of Variance")
abline(h = 1/p, col = "firebrick", lty = 2)

# ---- (b) UMAP on first 30 PCs ----
cat("\n=== Part (b): UMAP on First 30 PCs ===\n")
pca_30 <- pca_result$x[, 1:30]

set.seed(42)
umap_result <- umap(pca_30, n_neighbors = 15, min_dist = 0.1, verbose = FALSE)

subtype_labels <- factor(subtype, labels = c("Subtype 1", "Subtype 2",
                                              "Subtype 3", "Subtype 4 (rare)"))
cols <- c("steelblue", "firebrick", "forestgreen", "goldenrod")

plot_df <- tibble(
  UMAP1 = umap_result[, 1],
  UMAP2 = umap_result[, 2],
  Subtype = subtype_labels
)

# UMAP plot
par(mfrow = c(1, 1))
plot(umap_result, col = cols[subtype], pch = 16, cex = 0.7,
     main = "UMAP of Simulated Genomic Data (4 Cancer Subtypes)",
     xlab = "UMAP 1", ylab = "UMAP 2")
legend("topright", levels(subtype_labels), col = cols, pch = 16, cex = 0.8)

cat("The four subtypes are visually separable in the UMAP embedding.\n")
cat("Each subtype forms a distinct cluster, reflecting the different\n")
cat("gene expression profiles we simulated.\n")

# ---- (c) t-SNE comparison ----
cat("\n=== Part (c): t-SNE Comparison ===\n")

set.seed(42)
tsne_result <- Rtsne(pca_30, perplexity = 30, dims = 2,
                      verbose = FALSE, max_iter = 1000)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

plot(umap_result, col = cols[subtype], pch = 16, cex = 0.7,
     main = "UMAP (n_neighbors=15)", xlab = "UMAP 1", ylab = "UMAP 2")
legend("topright", levels(subtype_labels), col = cols, pch = 16, cex = 0.6)

plot(tsne_result$Y, col = cols[subtype], pch = 16, cex = 0.7,
     main = "t-SNE (perplexity=30)", xlab = "t-SNE 1", ylab = "t-SNE 2")
legend("topright", levels(subtype_labels), col = cols, pch = 16, cex = 0.6)

cat("Both UMAP and t-SNE successfully separate the four subtypes.\n")
cat("UMAP tends to preserve more global structure (relative distances\n")
cat("between clusters are more meaningful), while t-SNE may produce\n")
cat("more compact and well-separated clusters but with unreliable\n")
cat("inter-cluster distances.\n")

# ---- (d) Can the rare subtype be identified? ----
cat("\n=== Part (d): Identifying the Rare Subtype ===\n")

n_rare <- sum(subtype == 4)
cat("Rare subtype (Subtype 4) has", n_rare, "patients (5% of total).\n\n")

cat("YES, the rare subtype can be identified in the UMAP plot.\n")
cat("Despite having only ~50 patients, Subtype 4 forms a distinct\n")
cat("cluster (shown in goldenrod/yellow). This is because:\n")
cat("  1. The signal is strong (effect size = 3.5 in 40 genes)\n")
cat("  2. UMAP preserves local structure well, so even small groups\n")
cat("     remain cohesive\n")
cat("  3. The subtype's gene expression pattern is qualitatively\n")
cat("     different from the other subtypes (different genes)\n\n")
cat("In practice, rare subtypes CAN be identified if:\n")
cat("  - Their molecular profile is sufficiently distinct\n")
cat("  - The sample size is not too small (>20-30 patients)\n")
cat("  - Dimensionality reduction preserves the relevant structure\n")
cat("If the signal were weaker, the rare subtype might merge with\n")
cat("another group or be scattered as outliers.\n")
