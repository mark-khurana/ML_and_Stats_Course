# Generate simulated datasets for the course
# Run this script once to create the CSV files in data/

set.seed(42)

# ---- Simulated Meningitis Data ----
# Based on summary statistics from Lopez-Ayala et al. BMJ 2025
# and the Duke University Medical Center meningitis cohort

n <- 501
meningitis <- data.frame(
  id = 1:n,
  age = round(pmax(0, rnorm(n, mean = 40, sd = 20))),
  sex = sample(0:1, n, replace = TRUE, prob = c(0.45, 0.55))
)

# Bacterial meningitis (~45% prevalence in this dataset)
logit_p <- -1.5 +
  -0.02 * (meningitis$age - 40) +
  0.3 * meningitis$sex

# CSF glucose has a nonlinear relationship with bacterial meningitis
meningitis$csf_glucose <- round(pmax(5, rnorm(n, mean = 65, sd = 30)), 1)
# Low AND very high glucose increase bacterial probability (nonlinear)
glucose_effect <- -0.03 * meningitis$csf_glucose +
  0.0002 * (meningitis$csf_glucose - 50)^2

logit_p <- logit_p + glucose_effect

# Other CSF markers
meningitis$csf_leuk <- round(pmax(0, rlnorm(n, meanlog = 5, sdlog = 2)))
meningitis$csf_protein <- round(pmax(10, rnorm(n, mean = 120, sd = 80)), 1)
meningitis$blood_glucose <- round(pmax(50, rnorm(n, mean = 100, sd = 25)), 1)

# Higher leucocytes and protein increase bacterial probability
logit_p <- logit_p + 0.0003 * meningitis$csf_leuk + 0.005 * meningitis$csf_protein

prob_bacterial <- 1 / (1 + exp(-logit_p))
meningitis$bacterial <- rbinom(n, 1, prob_bacterial)

write.csv(meningitis, "data/meningitis_sim.csv", row.names = FALSE)
cat("Meningitis data:", nrow(meningitis), "rows,",
    sum(meningitis$bacterial), "bacterial cases\n")

# ---- Framingham-like simulated data ----
# Since real Framingham data requires registration, we simulate
# data with similar distributions for the course exercises

n_fram <- 4400
framingham <- data.frame(
  age = round(runif(n_fram, 32, 70)),
  sex = sample(0:1, n_fram, replace = TRUE),
  bmi = round(pmax(15, rnorm(n_fram, mean = 26, sd = 4.5)), 1),
  sbp = round(pmax(90, rnorm(n_fram, mean = 133, sd = 22))),
  dbp = round(pmax(50, rnorm(n_fram, mean = 83, sd = 12))),
  totchol = round(pmax(120, rnorm(n_fram, mean = 237, sd = 44))),
  hdl = round(pmax(20, rnorm(n_fram, mean = 50, sd = 15))),
  glucose = round(pmax(50, rnorm(n_fram, mean = 82, sd = 24))),
  smoking = sample(0:1, n_fram, replace = TRUE, prob = c(0.55, 0.45)),
  diabetes = sample(0:1, n_fram, replace = TRUE, prob = c(0.97, 0.03)),
  bp_meds = sample(0:1, n_fram, replace = TRUE, prob = c(0.92, 0.08))
)

# Generate 10-year CHD outcome
logit_chd <- -8.5 +
  0.065 * framingham$age +
  0.5 * framingham$sex +
  0.01 * framingham$sbp +
  0.005 * framingham$totchol +
  -0.02 * framingham$hdl +
  0.6 * framingham$smoking +
  0.8 * framingham$diabetes +
  0.3 * framingham$bp_meds

prob_chd <- 1 / (1 + exp(-logit_chd))
framingham$chd_10yr <- rbinom(n_fram, 1, prob_chd)
framingham$time_chd <- round(pmin(3650,
  rexp(n_fram, rate = 0.001) * ifelse(framingham$chd_10yr == 1, 0.5, 1)))

write.csv(framingham, "data/framingham.csv", row.names = FALSE)
cat("Framingham data:", nrow(framingham), "rows,",
    sum(framingham$chd_10yr), "CHD events\n")

cat("\nDatasets generated successfully!\n")
