# =============================================================================
# Chapter 8, Exercise 1: Build and Prune a Classification Tree
# 1. Fit a full, unpruned tree and count terminal nodes.
# 2. Use cross-validation to find the optimal cp.
# 3. Prune the tree and plot it.
# 4. Identify the top 3 splitting variables.
# =============================================================================

library(tidyverse)
library(rpart)
library(rpart.plot)

# --- Simulate the readmission dataset (same as chapter) ---
set.seed(42)
n <- 1000

readmit_data <- tibble(
  age = rnorm(n, 68, 12),
  length_of_stay = rpois(n, 5) + 1,
  num_comorbidities = rpois(n, 3),
  prior_admissions = rpois(n, 1),
  discharge_hemoglobin = rnorm(n, 11, 2),
  discharge_creatinine = rlnorm(n, 0.2, 0.5),
  has_diabetes = rbinom(n, 1, 0.35),
  has_chf = rbinom(n, 1, 0.25),
  readmitted = factor(
    rbinom(n, 1, plogis(-3 + 0.02 * (rnorm(n, 68, 12) - 68) +
                          0.15 * rpois(n, 1) +
                          0.1 * rpois(n, 3) +
                          0.3 * rbinom(n, 1, 0.25) -
                          0.1 * rnorm(n, 11, 2))),
    labels = c("No", "Yes")
  )
)

cat("Readmission rate:", mean(readmit_data$readmitted == "Yes"), "\n")

# --- Part 1: Fit a full, unpruned tree ---
full_tree <- rpart(readmitted ~ ., data = readmit_data, method = "class",
                   control = rpart.control(cp = 0.001, minsplit = 2, minbucket = 1))

n_terminal_full <- sum(full_tree$frame$var == "<leaf>")
cat("\nFull tree terminal nodes:", n_terminal_full, "\n")

# --- Part 2: Cross-validation to find optimal cp ---
cat("\nCP Table:\n")
printcp(full_tree)

# Plot CV error vs cp
plotcp(full_tree)

# Find optimal cp (minimum xerror)
cp_table <- full_tree$cptable
optimal_cp <- cp_table[which.min(cp_table[, "xerror"]), "CP"]
cat("\nOptimal cp:", optimal_cp, "\n")

# Alternative: 1-SE rule (smallest tree within 1 SE of the minimum)
min_xerror <- min(cp_table[, "xerror"])
min_se <- cp_table[which.min(cp_table[, "xerror"]), "xstd"]
cp_1se <- cp_table[cp_table[, "xerror"] <= min_xerror + min_se, "CP"]
optimal_cp_1se <- max(cp_1se)  # largest cp (smallest tree) within 1 SE
cat("Optimal cp (1-SE rule):", optimal_cp_1se, "\n")

# --- Part 3: Prune and plot ---
pruned_tree <- prune(full_tree, cp = optimal_cp)
n_terminal_pruned <- sum(pruned_tree$frame$var == "<leaf>")
cat("\nPruned tree terminal nodes:", n_terminal_pruned, "\n")

rpart.plot(pruned_tree, type = 4, extra = 106, under = TRUE,
           box.palette = "RdYlGn", roundint = FALSE,
           main = "Pruned Decision Tree: 30-Day Readmission")

# --- Part 4: Top 3 splitting variables ---
cat("\nVariable Importance:\n")
vi <- sort(full_tree$variable.importance, decreasing = TRUE)
print(vi)

top3 <- names(vi)[1:min(3, length(vi))]
cat("\nTop 3 splitting variables:", paste(top3, collapse = ", "), "\n")
cat("\nClinical interpretation:\n")
cat("- These variables capture patient acuity and complexity.\n")
cat("- Discharge lab values (hemoglobin, creatinine) reflect the patient's\n")
cat("  clinical status at the time of discharge.\n")
cat("- Age, comorbidity count, and prior admissions reflect overall\n")
cat("  disease burden and frailty.\n")
cat("- These are well-established risk factors for 30-day readmission\n")
cat("  in the clinical literature.\n")
