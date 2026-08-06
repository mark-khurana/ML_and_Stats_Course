# Chapter 13, Exercise 1: Diagnostic Test Updating
# Sequential Bayesian updating with mammogram and ultrasound

# ---- Part (a): Posterior probability after positive mammogram ----
cat("=== Part (a): Positive Mammogram ===\n")

sens_mammo <- 0.90   # Sensitivity of mammogram
spec_mammo <- 0.92   # Specificity of mammogram
prevalence <- 0.02   # Prior probability (prevalence in women aged 50-59)

# Apply Bayes' theorem:
# P(cancer | +) = P(+ | cancer) * P(cancer) / P(+)
# P(+) = P(+ | cancer)*P(cancer) + P(+ | no cancer)*P(no cancer)

p_pos <- sens_mammo * prevalence + (1 - spec_mammo) * (1 - prevalence)
post_mammo <- (sens_mammo * prevalence) / p_pos

cat("Prior (prevalence): ", prevalence, "\n")
cat("P(positive test):   ", round(p_pos, 4), "\n")
cat("P(cancer | positive mammogram):", round(post_mammo, 4), "\n")
cat("\nDespite 90% sensitivity and 92% specificity, the posterior\n")
cat("probability is only about", round(post_mammo * 100, 1), "% because\n")
cat("the prior probability (prevalence) is very low at 2%.\n")

# ---- Part (b): Sequential update with positive ultrasound ----
cat("\n=== Part (b): Sequential Update with Positive Ultrasound ===\n")

sens_us <- 0.95   # Sensitivity of ultrasound
spec_us <- 0.85   # Specificity of ultrasound

# The posterior from (a) becomes the new prior
prior_us <- post_mammo

p_pos_us <- sens_us * prior_us + (1 - spec_us) * (1 - prior_us)
post_us <- (sens_us * prior_us) / p_pos_us

cat("Prior (posterior from mammogram):", round(prior_us, 4), "\n")
cat("P(cancer | positive mammogram AND positive ultrasound):",
    round(post_us, 4), "\n")
cat("\nAfter two positive tests, the posterior probability rises to about",
    round(post_us * 100, 1), "%.\n")

# ---- Part (c): What does this illustrate? ----
cat("\n=== Part (c): What Sequential Updating Illustrates ===\n")
cat("1. NATURAL SEQUENTIAL UPDATING: In the Bayesian framework,\n")
cat("   today's posterior becomes tomorrow's prior. Each new piece\n")
cat("   of evidence updates our belief incrementally.\n\n")
cat("2. THE PRIOR MATTERS: Starting from a low prevalence (2%),\n")
cat("   even a sensitive test only raises the probability to ~19%.\n")
cat("   A second positive test raises it further to ~", round(post_us * 100), "%.\n\n")
cat("3. CLINICAL REASONING IS BAYESIAN: Clinicians intuitively do\n")
cat("   this -- they order confirmatory tests precisely because a\n")
cat("   single screening test in a low-prevalence population is\n")
cat("   insufficient. The Bayesian framework formalises this logic.\n\n")
cat("4. COHERENCE: The Bayesian approach naturally handles sequential\n")
cat("   evidence without the multiple-testing corrections that\n")
cat("   frequentist methods would require.\n")

# ---- Visualise the updating process ----
stages <- c("Prior\n(prevalence)", "After\nmammogram (+)", "After\nultrasound (+)")
probs <- c(prevalence, post_mammo, post_us)

barplot(probs, names.arg = stages, col = c("steelblue", "goldenrod", "firebrick"),
        main = "Sequential Bayesian Updating\nBreast Cancer Diagnosis",
        ylab = "P(Cancer)", ylim = c(0, max(probs) * 1.2),
        border = NA)
text(x = c(0.7, 1.9, 3.1), y = probs + 0.02,
     labels = paste0(round(probs * 100, 1), "%"), cex = 0.9)
