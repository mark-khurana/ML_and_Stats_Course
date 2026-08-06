# Chapter 11, Exercise 3: Decision Curve Interpretation
# Conceptual exercise about a preeclampsia prediction model

# This exercise is primarily conceptual. We simulate a plausible preeclampsia
# model and generate a decision curve to support the discussion.

set.seed(42)
n <- 2000

# Simulate data for preeclampsia prediction
preeclampsia_data <- data.frame(
  age = round(rnorm(n, 30, 5)),
  bmi = round(rnorm(n, 27, 5), 1),
  nulliparous = rbinom(n, 1, 0.4),
  history_pe = rbinom(n, 1, 0.05),
  map = round(rnorm(n, 85, 10))  # mean arterial pressure
)

lp <- -5.5 + 0.03 * preeclampsia_data$age +
  0.04 * preeclampsia_data$bmi +
  0.5 * preeclampsia_data$nulliparous +
  1.5 * preeclampsia_data$history_pe +
  0.02 * preeclampsia_data$map

preeclampsia_data$pe <- rbinom(n, 1, plogis(lp))
cat("Preeclampsia rate:", mean(preeclampsia_data$pe), "\n")

# Fit model
model <- glm(pe ~ age + bmi + nulliparous + history_pe + map,
             data = preeclampsia_data, family = binomial)

pred_prob <- predict(model, type = "response")
y_obs <- preeclampsia_data$pe

# ---- Generate decision curve ----
thresholds <- seq(0.01, 0.50, by = 0.01)
n_total <- length(y_obs)
nb_model <- nb_all <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  t <- thresholds[i]
  pred_pos <- as.numeric(pred_prob >= t)
  tp <- sum(pred_pos == 1 & y_obs == 1)
  fp <- sum(pred_pos == 1 & y_obs == 0)
  nb_model[i] <- tp / n_total - fp / n_total * (t / (1 - t))
  nb_all[i] <- mean(y_obs) - (1 - mean(y_obs)) * (t / (1 - t))
}

plot(thresholds, nb_model, type = "l", lwd = 2, col = "steelblue",
     main = "Decision Curve: Preeclampsia Prediction Model\n(AUC ~ 0.82)",
     xlab = "Threshold Probability", ylab = "Net Benefit",
     ylim = c(-0.05, max(c(nb_model, nb_all)) * 1.1))
lines(thresholds, nb_all, lwd = 2, col = "coral", lty = 2)
abline(h = 0, col = "black")
legend("topright", legend = c("Model", "Treat All", "Treat None"),
       col = c("steelblue", "coral", "black"),
       lty = c(1, 2, 1), lwd = 2)

# ---- Part (a): Range of useful thresholds ----
cat("\n=== Part (a): Range of Useful Thresholds ===\n")
cat("The model is useful at thresholds where its net benefit curve\n")
cat("exceeds BOTH the 'treat all' and 'treat none' lines.\n")
cat("From the decision curve, the model appears useful approximately\n")
cat("in the range of threshold probabilities from about 2% to 30-40%.\n")
cat("Below ~2%, 'treat all' has similar or higher net benefit.\n")
cat("Above ~30-40%, the model's net benefit approaches zero.\n")

# ---- Part (b): Counter-argument to 'AUC is only 0.82' ----
cat("\n=== Part (b): Counter-argument ===\n")
cat("An AUC of 0.82 is not 'only' -- it is strong discrimination.\n")
cat("More importantly, the AUC does not directly tell you whether\n")
cat("using the model improves clinical decisions.\n\n")
cat("The decision curve shows that across the clinically relevant\n")
cat("threshold range (e.g., 5-20%), the model provides positive\n")
cat("net benefit above both default strategies.\n\n")
cat("This means that using the model to guide closer monitoring\n")
cat("decisions would identify more true preeclampsia cases per\n")
cat("'unnecessary' monitoring than either monitoring everyone or\n")
cat("monitoring no one. Clinical utility is what matters for\n")
cat("patient care -- not the AUC alone.\n")

# ---- Part (c): Effect of action cost on the decision curve ----
cat("\n=== Part (c): Effect of Action Cost ===\n")
cat("The threshold probability reflects the implicit cost-benefit\n")
cat("ratio of the action (closer monitoring).\n\n")
cat("LOW-COST action (e.g., closer monitoring, extra visits):\n")
cat("  - Clinicians would use a LOW threshold (e.g., 3-5%)\n")
cat("  - They accept many false positives to avoid missing cases\n")
cat("  - The relevant region of the decision curve shifts LEFT\n")
cat("  - 'Treat all' remains competitive at low thresholds\n\n")
cat("HIGH-COST action (e.g., prophylactic medication with side effects):\n")
cat("  - Clinicians would use a HIGHER threshold (e.g., 15-25%)\n")
cat("  - They want more certainty before acting\n")
cat("  - The relevant region shifts RIGHT\n")
cat("  - The model has more value here because it avoids unnecessary\n")
cat("    treatment of low-risk patients\n\n")
cat("In summary: the decision curve itself does not change, but the\n")
cat("clinically relevant REGION changes depending on the cost of action.\n")
