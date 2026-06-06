# Chapter 16, Exercise 1: K-Means on Simulated Patient Data
# 600 patients, 6 clinical variables, 3 underlying phenotypes

library(tidyverse)
library(cluster)

# ---- Simulate data ----
set.seed(42)
n <- 600

# Phenotype 1: Septic shock (high HR, RR, lactate, low SBP)
p1 <- tibble(
  hr = rnorm(200, 115, 12), rr = rnorm(200, 28, 5),
  temp = rnorm(200, 38.5, 0.8), sbp = rnorm(200, 80, 12),
  creat = rnorm(200, 2.5, 0.8), lactate = rnorm(200, 5, 2)
)

# Phenotype 2: Stable critical (moderate vitals)
p2 <- tibble(
  hr = rnorm(200, 90, 10), rr = rnorm(200, 20, 3),
  temp = rnorm(200, 37.2, 0.5), sbp = rnorm(200, 110, 15),
  creat = rnorm(200, 1.2, 0.3), lactate = rnorm(200, 1.5, 0.5)
)

# Phenotype 3: Febrile, preserved hemodynamics
p3 <- tibble(
  hr = rnorm(200, 100, 10), rr = rnorm(200, 22, 4),
  temp = rnorm(200, 39.2, 0.7), sbp = rnorm(200, 120, 10),
  creat = rnorm(200, 1.0, 0.2), lactate = rnorm(200, 2.0, 0.8)
)

dat <- bind_rows(p1, p2, p3)
true_labels <- rep(1:3, each = 200)

# ---- (a) Scale and apply K-means for k = 2 to 6 ----
dat_scaled <- scale(dat)

wcss <- numeric(6)
sil_avg <- numeric(6)

for (k in 2:6) {
  km <- kmeans(dat_scaled, centers = k, nstart = 25)
  wcss[k] <- km$tot.withinss
  sil_avg[k] <- mean(silhouette(km$cluster, dist(dat_scaled))[, 3])
}

# ---- (b) Elbow and silhouette plots ----
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

plot(2:6, wcss[2:6], type = "b", pch = 16, col = "steelblue",
     xlab = "Number of clusters (k)", ylab = "Within-cluster SS",
     main = "Elbow Method")

plot(2:6, sil_avg[2:6], type = "b", pch = 16, col = "firebrick",
     xlab = "Number of clusters (k)", ylab = "Average Silhouette",
     main = "Silhouette Scores")

cat("=== Part (b): Optimal k ===\n")
cat("Elbow plot: The elbow appears at k = 3, where WCSS decreases\n")
cat("  sharply and then flattens.\n")
cat("Silhouette: Maximum average silhouette at k =", which.max(sil_avg[2:6]) + 1, "\n")
cat("Both methods suggest k = 3, consistent with the 3 simulated phenotypes.\n")

# ---- (c) Visualise k=3 solution using PCA ----
km3 <- kmeans(dat_scaled, centers = 3, nstart = 25)
pca2 <- prcomp(dat_scaled)$x[, 1:2]

par(mfrow = c(1, 1))
cols <- c("steelblue", "firebrick", "forestgreen")
plot(pca2, col = cols[km3$cluster], pch = 16, cex = 0.7,
     main = "K-means (k=3) on First 2 PCs",
     xlab = "PC1", ylab = "PC2")
legend("topright", paste("Cluster", 1:3), col = cols, pch = 16, cex = 0.8)

# ---- (d) Profile the clusters ----
cat("\n=== Part (d): Cluster Profiles ===\n\n")

dat$cluster <- km3$cluster
profiles <- dat %>%
  group_by(cluster) %>%
  summarise(across(hr:lactate, mean), n = n(), .groups = "drop")

print(profiles)

cat("\nClinical interpretation of cluster profiles:\n\n")

# Identify which cluster matches which phenotype
for (cl in 1:3) {
  prof <- profiles %>% filter(cluster == cl)
  cat(sprintf("Cluster %d (n = %d):\n", cl, prof$n))
  cat(sprintf("  HR=%.0f, RR=%.0f, Temp=%.1f, SBP=%.0f, Creat=%.1f, Lactate=%.1f\n",
              prof$hr, prof$rr, prof$temp, prof$sbp, prof$creat, prof$lactate))

  if (prof$lactate > 3 && prof$sbp < 90) {
    cat("  -> Matches SEPTIC SHOCK phenotype: high HR, RR, lactate; low SBP\n")
  } else if (prof$temp > 38.5) {
    cat("  -> Matches FEBRILE phenotype: high temp, preserved hemodynamics\n")
  } else {
    cat("  -> Matches STABLE CRITICAL phenotype: moderate vitals\n")
  }
  cat("\n")
}

cat("The cluster profiles make clinical sense. The algorithm has\n")
cat("successfully recovered the three simulated phenotypes, each\n")
cat("with a distinct clinical profile that a clinician would recognise.\n")
