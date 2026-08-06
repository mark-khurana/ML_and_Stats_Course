# Chapter 16, Exercise 3: DBSCAN for Outlier Detection
# Add outlier patients to the Exercise 1 dataset

library(tidyverse)
library(cluster)
library(dbscan)

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
is_outlier <- rep(FALSE, 600)

# Add 30 outlier patients with extreme values
set.seed(99)
outliers <- tibble(
  hr = rnorm(30, 180, 10),       # Extreme heart rate
  rr = rnorm(30, 40, 5),         # Extreme respiratory rate
  temp = rnorm(30, 40.5, 0.5),   # Very high temperature
  sbp = rnorm(30, 50, 10),       # Very low blood pressure
  creat = rnorm(30, 8, 1.5),     # Very high creatinine
  lactate = rnorm(30, 15, 3)     # Very high lactate
)

dat <- bind_rows(dat, outliers)
is_outlier <- c(is_outlier, rep(TRUE, 30))
dat_scaled <- scale(dat)

cat("Total patients:", nrow(dat), "(600 normal + 30 outliers)\n\n")

# ---- (a) K-means with k=3 ----
cat("=== Part (a): K-means with k=3 ===\n")

km3 <- kmeans(dat_scaled, centers = 3, nstart = 25)

# Where do outliers end up?
outlier_clusters <- km3$cluster[is_outlier]
cat("Outlier distribution across K-means clusters:\n")
print(table(outlier_clusters))

cat("\nK-means assigns outliers to one of the existing clusters,\n")
cat("typically the cluster whose centroid is nearest (even if far).\n")
cat("This can distort the centroid and corrupt the cluster profiles.\n")

# Show how outliers affect cluster means
dat$km_cluster <- km3$cluster
dat$is_outlier <- is_outlier

for (cl in 1:3) {
  n_outlier_in_cl <- sum(dat$km_cluster == cl & dat$is_outlier)
  n_total_in_cl <- sum(dat$km_cluster == cl)
  cat(sprintf("Cluster %d: %d patients (%d outliers)\n",
              cl, n_total_in_cl, n_outlier_in_cl))
}

# ---- (b) DBSCAN ----
cat("\n=== Part (b): DBSCAN ===\n")

# Use kNNdistplot to help choose eps
kNNdistplot(dat_scaled, k = 5)
abline(h = 2.5, col = "firebrick", lty = 2)

# Run DBSCAN
db <- dbscan(dat_scaled, eps = 2.5, minPts = 10)

cat("DBSCAN results:\n")
cat("  Number of clusters:", max(db$cluster), "\n")
cat("  Noise points (outliers):", sum(db$cluster == 0), "\n")
cat("  Actual outliers detected as noise:",
    sum(db$cluster == 0 & is_outlier), "out of 30\n")

cat("\nDBSCAN cluster sizes:\n")
print(table(db$cluster))

# ---- (c) Compare non-outlier assignments ----
cat("\n=== Part (c): Comparison for Non-Outlier Patients ===\n")

# Get cluster assignments for non-outlier patients only
normal_idx <- !is_outlier
km_normal <- km3$cluster[normal_idx]
db_normal <- db$cluster[normal_idx]

# For DBSCAN, remove any normal patients assigned as noise
db_noise_normal <- sum(db_normal == 0)
cat("Normal patients misclassified as noise by DBSCAN:", db_noise_normal, "\n")

# ARI for non-noise, non-outlier patients
both_assigned <- db_normal > 0
if (sum(both_assigned) > 0) {
  # Use mclust for ARI
  library(mclust)
  ari <- adjustedRandIndex(km_normal[both_assigned], db_normal[both_assigned])
  cat("ARI between K-means and DBSCAN (non-outlier, non-noise):", round(ari, 3), "\n")
}

# ---- (d) Which approach for outlier handling? ----
cat("\n=== Part (d): Preferred Approach for Outlier Handling ===\n\n")

cat("DBSCAN is preferred for clinical phenotyping when outliers are expected.\n\n")

cat("Reasons:\n")
cat("1. EXPLICIT OUTLIER IDENTIFICATION: DBSCAN labels outliers as noise,\n")
cat("   making them visible for clinical review. K-means silently absorbs\n")
cat("   them into clusters, distorting the results.\n\n")

cat("2. CLINICAL RELEVANCE: Outlier patients (e.g., with extreme vitals)\n")
cat("   may represent data errors, rare presentations, or patients who\n")
cat("   need individual assessment. Identifying them is valuable.\n\n")

cat("3. CLUSTER INTEGRITY: By excluding outliers, DBSCAN preserves the\n")
cat("   purity of the main clusters. K-means cluster centroids can be\n")
cat("   pulled by outliers, producing misleading profiles.\n\n")

cat("4. PRACTICAL WORKFLOW:\n")
cat("   - Use DBSCAN to identify outliers first.\n")
cat("   - Review outliers clinically (data errors? rare cases?).\n")
cat("   - Apply K-means to the remaining non-outlier patients for\n")
cat("     cleaner phenotyping.\n\n")

cat("5. CAVEAT: DBSCAN requires tuning eps and minPts, which can be\n")
cat("   challenging. The kNN distance plot helps, but the choice\n")
cat("   affects how many points are labelled as noise.\n")

# Visualise comparison
pca2 <- prcomp(dat_scaled)$x[, 1:2]

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# K-means
cols_km <- c("steelblue", "firebrick", "forestgreen")[km3$cluster]
pch_km <- ifelse(is_outlier, 4, 16)
plot(pca2, col = cols_km, pch = pch_km, cex = 0.7,
     main = "K-means (k=3)\n(X = outlier patients)",
     xlab = "PC1", ylab = "PC2")

# DBSCAN
db_cols <- c("grey50", "steelblue", "firebrick", "forestgreen")
cols_db <- db_cols[db$cluster + 1]
pch_db <- ifelse(db$cluster == 0, 4, 16)
plot(pca2, col = cols_db, pch = pch_db, cex = 0.7,
     main = "DBSCAN\n(X/grey = noise)",
     xlab = "PC1", ylab = "PC2")
legend("topright", c("Noise", "Cluster 1", "Cluster 2", "Cluster 3"),
       col = db_cols, pch = c(4, 16, 16, 16), cex = 0.7)
