# Chapter 15, Exercise 1: PCA on Clinical Lab Data
# Using the simulated metabolic panel data from the chapter

library(tidyverse)
library(MASS)

# ---- Simulate metabolic data (same as chapter) ----
set.seed(42)
n <- 300

Sigma <- matrix(c(
  1.0, 0.7, 0.5, 0.3,-0.2, 0.1, 0.2,-0.1,
  0.7, 1.0, 0.4, 0.2,-0.2, 0.1, 0.1,-0.1,
  0.5, 0.4, 1.0, 0.4,-0.3, 0.1, 0.3,-0.1,
  0.3, 0.2, 0.4, 1.0,-0.5, 0.1, 0.2, 0.0,
 -0.2,-0.2,-0.3,-0.5, 1.0,-0.1,-0.1, 0.2,
  0.1, 0.1, 0.1, 0.1,-0.1, 1.0, 0.1,-0.3,
  0.2, 0.1, 0.3, 0.2,-0.1, 0.1, 1.0,-0.2,
 -0.1,-0.1,-0.1, 0.0, 0.2,-0.3,-0.2, 1.0
), nrow = 8)

z <- mvrnorm(n, mu = rep(0, 8), Sigma = Sigma)
metabolic <- tibble(
  glucose       = round(z[,1] * 30 + 100),
  hba1c         = round(z[,2] * 1.0 + 5.8, 1),
  triglycerides = round(z[,3] * 50 + 150),
  ldl           = round(z[,4] * 30 + 120),
  hdl           = round(z[,5] * 12 + 55),
  creatinine    = round(z[,6] * 0.3 + 1.0, 2),
  alt           = round(z[,7] * 15 + 30),
  albumin       = round(z[,8] * 0.4 + 4.0, 1)
)

# ---- (a) Perform PCA on standardised data ----
cat("=== Part (a): PCA on Standardised Data ===\n")
pca_fit <- prcomp(metabolic, scale. = TRUE)
cat("PCA completed. Summary:\n")
print(summary(pca_fit))

# ---- (b) Scree plot and number of components ----
cat("\n=== Part (b): Scree Plot and Component Selection ===\n")

var_prop <- pca_fit$sdev^2 / sum(pca_fit$sdev^2)
cum_var  <- cumsum(var_prop)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Scree plot
barplot(var_prop, names.arg = paste0("PC", 1:8), col = "steelblue",
        main = "Scree Plot", ylab = "Proportion of Variance",
        xlab = "Component", ylim = c(0, 0.4))
abline(h = 1/8, lty = 2, col = "firebrick")
text(9, 1/8 + 0.015, "Kaiser threshold (1/p)", col = "firebrick", cex = 0.8)

# Cumulative variance
plot(1:8, cum_var, type = "b", pch = 16, col = "steelblue",
     main = "Cumulative Variance", xlab = "Number of Components",
     ylab = "Cumulative Proportion", ylim = c(0, 1))
abline(h = 0.80, col = "firebrick", lty = 2)
text(6, 0.82, "80% threshold", col = "firebrick", cex = 0.8)

# Kaiser criterion: eigenvalues > 1 (which equals variance > 1/p for scaled data)
eigenvalues <- pca_fit$sdev^2
n_kaiser <- sum(eigenvalues > 1)
n_80 <- which(cum_var >= 0.80)[1]

cat("\nKaiser criterion (eigenvalue > 1):", n_kaiser, "components\n")
cat("80% cumulative variance threshold:", n_80, "components\n")
cat("Eigenvalues:", round(eigenvalues, 3), "\n")
cat("Cumulative variance:", round(cum_var, 3), "\n")

# ---- (c) Loadings of first two PCs ----
cat("\n=== Part (c): Loadings and Interpretation ===\n")

cat("\nPC1 loadings (sorted by absolute value):\n")
pc1_loadings <- pca_fit$rotation[, 1]
print(round(sort(abs(pc1_loadings), decreasing = TRUE), 3))
cat("Signed loadings:\n")
print(round(pc1_loadings[order(abs(pc1_loadings), decreasing = TRUE)], 3))

cat("\nPC2 loadings (sorted by absolute value):\n")
pc2_loadings <- pca_fit$rotation[, 2]
print(round(sort(abs(pc2_loadings), decreasing = TRUE), 3))
cat("Signed loadings:\n")
print(round(pc2_loadings[order(abs(pc2_loadings), decreasing = TRUE)], 3))

cat("\nClinical interpretation:\n")
cat("PC1: Dominated by glucose, HbA1c, triglycerides (positive) and\n")
cat("     HDL (negative). This is a 'metabolic syndrome' axis.\n")
cat("     Patients scoring high on PC1 tend to have elevated glucose,\n")
cat("     HbA1c, triglycerides and lower HDL.\n\n")
cat("PC2: Dominated by creatinine and albumin (with opposite signs).\n")
cat("     This captures a 'renal/hepatic function' dimension.\n")
cat("     Patients with high creatinine and low albumin (suggesting\n")
cat("     renal impairment or liver dysfunction) score high on PC2.\n")

# ---- (d) Biplot coloured by diabetes status ----
cat("\n=== Part (d): Biplot with Diabetes Status ===\n")

set.seed(99)
metabolic$diabetes <- factor(
  ifelse(metabolic$glucose > 110 & metabolic$hba1c > 6.2,
         "Diabetic", "Non-diabetic")
)

cat("Diabetes prevalence:", mean(metabolic$diabetes == "Diabetic"), "\n")

# Extract PC scores
pc_scores <- as.data.frame(pca_fit$x[, 1:2])
pc_scores$diabetes <- metabolic$diabetes

par(mfrow = c(1, 1))

# Biplot with diabetes colouring
cols <- ifelse(metabolic$diabetes == "Diabetic", "firebrick", "steelblue")
plot(pc_scores$PC1, pc_scores$PC2, col = cols, pch = 16, cex = 0.7,
     xlab = paste0("PC1 (", round(var_prop[1] * 100, 1), "% var)"),
     ylab = paste0("PC2 (", round(var_prop[2] * 100, 1), "% var)"),
     main = "PCA Biplot Coloured by Diabetes Status")

# Add loading arrows
loadings <- pca_fit$rotation[, 1:2]
scale_factor <- 3
for (i in 1:nrow(loadings)) {
  arrows(0, 0, loadings[i, 1] * scale_factor, loadings[i, 2] * scale_factor,
         col = "grey30", length = 0.1, lwd = 1.5)
  text(loadings[i, 1] * scale_factor * 1.15,
       loadings[i, 2] * scale_factor * 1.15,
       rownames(loadings)[i], cex = 0.7, col = "grey20")
}

legend("topright", c("Diabetic", "Non-diabetic"),
       col = c("firebrick", "steelblue"), pch = 16, cex = 0.8)

cat("\nDiabetic patients tend to cluster toward higher PC1 values,\n")
cat("which aligns with the 'metabolic syndrome' interpretation.\n")
cat("This confirms that PC1 captures the metabolic dimension\n")
cat("along which diabetic patients differ from non-diabetic patients.\n")
