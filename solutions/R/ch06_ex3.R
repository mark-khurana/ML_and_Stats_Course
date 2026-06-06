# =============================================================================
# Chapter 6, Exercise 3: Competing Risks Thinking
# Kidney transplant rejection with death as a competing risk
# =============================================================================
# This exercise is primarily conceptual (parts 1-2) with a bonus coding
# component (part 3).

# --- Question 1 ---
# Explain why treating death as censoring would bias the results.
# What direction would the bias go?
#
# ANSWER:
# Treating death as simple censoring VIOLATES the assumption of
# NON-INFORMATIVE censoring. Non-informative censoring means that
# censored patients have the same future risk of the event as those
# who remain under observation. But patients who die are NOT like
# patients who remain alive -- they have ZERO future risk of rejection
# (because dead patients cannot experience rejection).
#
# The bias direction: treating death as censoring INFLATES (overestimates)
# the estimated probability of rejection. This happens because the
# Kaplan-Meier estimator assumes that censored patients would eventually
# experience the event at the same rate as those still at risk. But dead
# patients will NEVER experience rejection, so acting as if they could
# leads to an overestimate of the cumulative incidence of rejection.
#
# In the KM framework, when a patient dies and is "censored," they are
# removed from the risk set, effectively assuming they would have had
# the same rejection rate as surviving patients. Since they actually
# have zero rejection risk, this inflates the estimated rate.

# --- Question 2 ---
# Which approach for estimating probability of rejection within 2 years:
# cause-specific hazards or Fine-Gray?
#
# ANSWER:
# For estimating the PROBABILITY of rejection within 2 years, use the
# FINE-GRAY subdistribution hazard model. Here is why:
#
# - Cause-specific hazards model the RATE of rejection among those
#   currently alive and rejection-free. This answers: "Among patients
#   still alive, what is the instantaneous risk of rejection?" This is
#   useful for understanding ETIOLOGY (what causes rejection).
#
# - The Fine-Gray model directly models the CUMULATIVE INCIDENCE
#   FUNCTION, which gives the probability of rejection by time t,
#   accounting for the fact that some patients will die first. This
#   is the right quantity for PREDICTION and CLINICAL COMMUNICATION.
#
# When your goal is prediction ("What is the probability that this
# patient's transplant will be rejected within 2 years?"), you need
# the cumulative incidence, which properly accounts for the competing
# risk of death. The Fine-Gray model provides this directly.

# --- Question 3 (Bonus): Fine-Gray model in R ---

library(survival)
library(tidycmprsk)
library(ggplot2)

# Simulate kidney transplant data with competing risks
set.seed(2025)
n <- 500

# Covariates
age <- rnorm(n, mean = 50, sd = 12)
donor_type <- rbinom(n, 1, 0.4)  # 0 = living, 1 = deceased donor
hla_mismatch <- rpois(n, lambda = 2)

# Simulate competing event times
# Time to rejection (cause 1)
lambda_reject <- exp(-4 + 0.02 * age + 0.5 * donor_type + 0.2 * hla_mismatch)
time_reject <- rexp(n, rate = lambda_reject)

# Time to death without rejection (cause 2)
lambda_death <- exp(-5 + 0.03 * age + 0.3 * donor_type)
time_death <- rexp(n, rate = lambda_death)

# Administrative censoring at 10 years
time_censor <- runif(n, 5, 10)

# Determine observed time and event type
time_obs <- pmin(time_reject, time_death, time_censor)
event <- ifelse(time_obs == time_censor, 0,
                ifelse(time_obs == time_reject, 1, 2))
# 0 = censored, 1 = rejection, 2 = death

df <- data.frame(
  time = time_obs,
  event = event,
  age = age,
  donor_type = factor(donor_type, labels = c("Living", "Deceased")),
  hla_mismatch = hla_mismatch
)

cat("=== Dataset summary ===\n")
cat("N:", n, "\n")
cat("Rejections (event=1):", sum(event == 1), "\n")
cat("Deaths (event=2):", sum(event == 2), "\n")
cat("Censored (event=0):", sum(event == 0), "\n\n")

# --- Cumulative Incidence Function (CIF) ---
# Using tidycmprsk for a tidy interface
cuminc_fit <- cuminc(Surv(time, event) ~ donor_type, data = df)
print(cuminc_fit)

# Plot cumulative incidence by donor type
p <- ggcuminc(cuminc_fit, outcome = "1") +  # outcome 1 = rejection
  labs(x = "Time (years)",
       y = "Cumulative Incidence of Rejection",
       title = "Cumulative Incidence of Transplant Rejection by Donor Type") +
  theme_minimal(base_size = 14)
print(p)

# --- Fine-Gray subdistribution hazard model ---
cat("\n=== Fine-Gray Model ===\n")
# Using tidycmprsk::crr for Fine-Gray regression
fg_model <- crr(Surv(time, event) ~ age + donor_type + hla_mismatch,
                data = df, failcode = 1)  # failcode 1 = rejection
print(summary(fg_model))

cat("\nInterpretation:\n")
cat("- The Fine-Gray model estimates subdistribution hazard ratios.\n")
cat("- These quantify how covariates affect the cumulative incidence\n")
cat("  of rejection, accounting for the competing risk of death.\n")
cat("- HR > 1 means higher cumulative incidence of rejection.\n")

# --- Compare with cause-specific Cox model ---
cat("\n=== Cause-Specific Cox Model (for comparison) ===\n")
# Censor deaths for cause-specific analysis of rejection
df$event_cs <- ifelse(df$event == 1, 1, 0)  # only rejection is event
cox_cs <- coxph(Surv(time, event_cs) ~ age + donor_type + hla_mismatch,
                data = df)
print(summary(cox_cs))

cat("\n=== Comparison Summary ===\n")
cat("The cause-specific and Fine-Gray HRs can differ because they\n")
cat("answer different questions:\n")
cat("- Cause-specific: 'What affects the rate of rejection among\n")
cat("  those still alive?'\n")
cat("- Fine-Gray: 'What affects the probability of rejection by\n")
cat("  time t, accounting for death?'\n")
cat("Both are valid; the choice depends on whether your goal is\n")
cat("etiological understanding or clinical prediction.\n")
