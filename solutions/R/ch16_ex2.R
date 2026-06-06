# Chapter 16, Exercise 2: Hierarchical Clustering with Different Linkage Methods
# Using the same dataset from Exercise 1

library(tidyverse)
library(cluster)
library(mclust)  # for adjustedRandIndex

# ---- Simulate data (same as Exercise 1) ----
set.seed(42)

p1 <- tibble(hr = rnorm(200, 115, 12), rr = rnorm(200, 28, 5),
             temp = rnorm(200, 38.5, 0.8), sbp = rnorm(200, 80, 12),
             creat = rnorm(200, 2.5, 0.8), lactate = rnorm(200, 5, 2))
p2 <- tibble(hr = rnorm(200, 90, 10), rr = rnorm(200, 20, 3),
             temp = rnorm(200, 37.2, 0.5), sbp = rnorm(200, 110, 15),
             creat = rnorm(200, 1.2, 0.3), lactate = rnorm(200, 1.5, 0.5))
p3 <- tibble(hr = rnorm(200, 100, 10), rr = rnorm(200, 22, 4),
             temp = rnorm(200, 39.2, 0.7), sbp = rnorm(200, 120, 10),
             creat = rnorm(200, 1.0, 0.2), lactate = rnorm(200, 2.0, 0.8))

dat <- bind_rows(p1, p2, p3)
dat_scaled <- scale(dat)

# K-means reference solution
km3 <- kmeans(dat_scaled, centers = 3, nstart = 25)

# ---- (a) Hierarchical clustering with 4 linkage methods ----
cat("=== Part (a): Hierarchical Clustering ===\n\n")

d <- dist(dat_scaled)
methods <- c("single", "complete", "average", "ward.D2")
titles <- c("Single", "Complete", "Average", "Ward's")

# Use a subset for readable dendrograms
set.seed(42)
idx <- sample(nrow(dat_scaled), 80)

par(mfrow = c(2, 2), mar = c(2, 3, 3, 1))
for (i in seq_along(methods)) {
  hc <- hclust(dist(dat_scaled[idx, ]), method = methods[i])
  plot(hc, labels = FALSE, main = titles[i], xlab = "", sub = "",
       hang = -1, cex = 0.5)
  rect.hclust(hc, k = 3, border = "firebrick")
}

# ---- (b) Cut each dendrogram at k=3, compare assignments ----
cat("=== Part (b): Cluster Assignments ===\n\n")

hc_clusters <- list()
for (i in seq_along(methods)) {
  hc <- hclust(d, method = methods[i])
  hc_clusters[[methods[i]]] <- cutree(hc, k = 3)
}

# Crosstabulation between methods
cat("Crosstab: Ward's vs Complete linkage:\n")
print(table(Ward = hc_clusters[["ward.D2"]],
            Complete = hc_clusters[["complete"]]))

cat("\nCrosstab: Ward's vs Single linkage:\n")
print(table(Ward = hc_clusters[["ward.D2"]],
            Single = hc_clusters[["single"]]))

# ---- (c) ARI comparison with K-means ----
cat("\n=== Part (c): Adjusted Rand Index vs K-means ===\n\n")

for (i in seq_along(methods)) {
  ari <- adjustedRandIndex(km3$cluster, hc_clusters[[methods[i]]])
  cat(sprintf("ARI(%s vs K-means): %.3f\n", titles[i], ari))
}

cat("\nWard's method produces clusters most similar to K-means (highest ARI).\n")
cat("This is expected because both Ward's method and K-means minimise\n")
cat("within-cluster variance (WCSS), so they have similar objectives.\n")

# ---- (d) Recommendation for clinical data ----
cat("\n=== Part (d): Recommendation ===\n\n")

cat("Ward's method is the recommended default for clinical data because:\n\n")
cat("1. OBJECTIVE: Ward's minimises within-cluster variance, which\n")
cat("   aligns with the goal of finding compact, homogeneous clusters.\n\n")
cat("2. CONSISTENCY: It produces results most similar to K-means,\n")
cat("   which is the most widely used method. Using both and checking\n")
cat("   for agreement strengthens confidence in the results.\n\n")
cat("3. ROBUSTNESS: Complete linkage can be distorted by outliers;\n")
cat("   single linkage creates chaining effects (long, straggling\n")
cat("   clusters). Ward's avoids both issues.\n\n")
cat("4. COMPACT CLUSTERS: Clinical phenotypes are typically expected\n")
cat("   to be compact groups of similar patients, which Ward's\n")
cat("   naturally produces.\n\n")
cat("5. INTERPRETABILITY: The dendrogram from Ward's method is\n")
cat("   usually the easiest to interpret, with clear height gaps\n")
cat("   indicating natural cluster boundaries.\n\n")
cat("CAVEAT: If you suspect non-spherical clusters (e.g., disease\n")
cat("trajectories that form elongated shapes), average linkage\n")
cat("might be more appropriate. But for most clinical phenotyping\n")
cat("applications, Ward's is the safe default.\n")
