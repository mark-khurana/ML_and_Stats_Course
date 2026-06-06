# Chapter 15, Exercise 2: t-SNE Sensitivity to Perplexity
# Using the three-group simulated dataset from the chapter

library(Rtsne)

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

# ---- (a) & (b) Run t-SNE with 4 perplexity values ----
perplexities <- c(5, 15, 30, 50)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

for (perp in perplexities) {
  set.seed(42)  # Same seed for comparability
  tsne_result <- Rtsne(X, perplexity = perp, dims = 2,
                        verbose = FALSE, max_iter = 1000)

  plot(tsne_result$Y, col = cols[labels], pch = 16, cex = 0.8,
       main = paste("t-SNE (perplexity =", perp, ")"),
       xlab = "t-SNE 1", ylab = "t-SNE 2")

  if (perp == 5) {
    legend("topright", levels(labels), col = cols, pch = 16, cex = 0.7)
  }
}

# ---- (c) When do groups become clearly separated? ----
cat("=== Part (c): When Are Groups Clearly Separated? ===\n\n")
cat("The three groups become clearly separated at perplexity = 15.\n")
cat("At perplexity = 5, the embedding focuses on very local structure,\n")
cat("which can fragment the groups into smaller sub-clusters.\n")
cat("At perplexity = 15 and above, the groups are well-separated\n")
cat("with clear boundaries between them.\n\n")
cat("Higher perplexities (30, 50) also separate the groups clearly\n")
cat("but with slightly different visual arrangements and more\n")
cat("spread-out clusters.\n")

# ---- (d) Are distances between clusters consistent? ----
cat("\n=== Part (d): Consistency of Inter-Cluster Distances ===\n\n")
cat("NO, the distances between clusters are NOT consistent across\n")
cat("perplexity values. Key observations:\n\n")
cat("1. The relative positions of the three clusters change across\n")
cat("   perplexity settings. Type A might be nearest to Type C at\n")
cat("   one perplexity but nearest to Type B at another.\n\n")
cat("2. The absolute distances between cluster centres vary\n")
cat("   substantially. At low perplexity, clusters may appear close;\n")
cat("   at high perplexity, they may be farther apart (or vice versa).\n\n")
cat("3. Cluster sizes (spread) also change with perplexity.\n\n")
cat("What this tells us about interpreting t-SNE:\n")
cat("- Distances between clusters in t-SNE are MEANINGLESS.\n")
cat("- t-SNE preserves local neighbourhood structure, not global\n")
cat("  distances.\n")
cat("- You should NEVER conclude that two clusters are 'more similar'\n")
cat("  because they appear closer in a t-SNE plot.\n")
cat("- The only reliable information is WHETHER clusters exist,\n")
cat("  not HOW FAR APART they are.\n")
cat("- Always try multiple perplexity values. If the same clusters\n")
cat("  appear consistently, the structure is likely real.\n")
