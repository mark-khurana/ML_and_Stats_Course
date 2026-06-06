# Chapter 15, Exercise 3: UMAP Parameter Exploration
# Using the three-group simulated dataset from the chapter

library(uwot)

# ---- Simulate data (same as chapter) ----
set.seed(42)
n_per_group <- 100
p <- 20

group1 <- matrix(rnorm(n_per_group * p, mean = 0, sd = 1), ncol = p)
group2 <- matrix(rnorm(n_per_group * p, mean = 2, sd = 1.2), ncol = p)
group3 <- matrix(rnorm(n_per_group * p, mean = -1, sd = 0.8), ncol = p)
group2[, 1:5] <- group2[, 1:5] + 3
group3[, 10:15] <- group3[, 10:15] - 4

X <- rbind(group1, group2, group3)
labels <- factor(rep(c("Type A", "Type B", "Type C"), each = n_per_group))
cols <- c("steelblue", "firebrick", "forestgreen")

# ---- (a) UMAP with varying n_neighbors, min_dist = 0.1 ----
n_neighbors_vals <- c(5, 15, 50, 100)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

for (nn in n_neighbors_vals) {
  set.seed(42)
  umap_result <- umap(X, n_neighbors = nn, min_dist = 0.1, verbose = FALSE)

  plot(umap_result, col = cols[labels], pch = 16, cex = 0.8,
       main = paste("n_neighbors =", nn, ", min_dist = 0.1"),
       xlab = "UMAP 1", ylab = "UMAP 2")

  if (nn == 5) {
    legend("topright", levels(labels), col = cols, pch = 16, cex = 0.7)
  }
}

# ---- (b) UMAP with varying min_dist, n_neighbors = 15 ----
min_dist_vals <- c(0.0, 0.1, 0.5, 1.0)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

for (md in min_dist_vals) {
  set.seed(42)
  umap_result <- umap(X, n_neighbors = 15, min_dist = md, verbose = FALSE)

  plot(umap_result, col = cols[labels], pch = 16, cex = 0.8,
       main = paste("n_neighbors = 15, min_dist =", md),
       xlab = "UMAP 1", ylab = "UMAP 2")

  if (md == 0.0) {
    legend("topright", levels(labels), col = cols, pch = 16, cex = 0.7)
  }
}

# ---- (c) Already done above via the two panels of plots ----

# ---- (d) Description and recommendation ----
cat("=== Part (d): How Each Parameter Affects Visual Appearance ===\n\n")

cat("n_neighbors (with min_dist = 0.1 fixed):\n")
cat("  n_neighbors = 5:   Very tight, fragmented clusters. Each point\n")
cat("    considers only 5 neighbors, emphasising micro-structure.\n")
cat("    May split true clusters into sub-groups.\n")
cat("  n_neighbors = 15:  Well-separated, compact clusters. Good\n")
cat("    balance between local and global structure.\n")
cat("  n_neighbors = 50:  Broader clusters, more connected. Groups\n")
cat("    start to merge slightly as the algorithm considers wider\n")
cat("    neighbourhoods.\n")
cat("  n_neighbors = 100: Even more spread out. Global structure is\n")
cat("    emphasised over local detail. Clusters are less tightly packed.\n\n")

cat("min_dist (with n_neighbors = 15 fixed):\n")
cat("  min_dist = 0.0:  Very tight, dense clusters. Points are packed\n")
cat("    as close together as possible. Maximum visual separation.\n")
cat("  min_dist = 0.1:  Slightly looser clusters. Good default.\n")
cat("  min_dist = 0.5:  More spread-out embedding. Internal cluster\n")
cat("    structure becomes more visible but inter-cluster gaps shrink.\n")
cat("  min_dist = 1.0:  Very spread out, almost uniform. Clusters\n")
cat("    overlap, and the embedding loses much of its structure.\n\n")

cat("Recommendation for this dataset:\n")
cat("  n_neighbors = 15, min_dist = 0.1\n")
cat("  This combination provides well-separated, compact clusters\n")
cat("  that clearly reveal the three-group structure. It is also\n")
cat("  the default in most implementations, and for good reason:\n")
cat("  it balances local detail with global structure effectively.\n")
