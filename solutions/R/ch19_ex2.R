# =============================================================================
# Chapter 19 - Exercise 2: Publication-Quality Multi-Panel Figure in R
# Survival analysis of the lung dataset
# =============================================================================

library(tidyverse)
library(survival)
library(survminer)
library(broom)
library(patchwork)

# --- Publication theme and colour-blind palette ---
theme_publication <- function(base_size = 11) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      text = element_text(family = "Arial"),
      plot.title = element_text(size = base_size + 2, face = "bold",
                                hjust = 0, margin = margin(b = 10)),
      axis.title = element_text(size = base_size, face = "bold"),
      axis.text = element_text(size = base_size - 1, colour = "black"),
      legend.title = element_text(size = base_size, face = "bold"),
      legend.text = element_text(size = base_size - 1),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey90"),
      strip.text = element_text(size = base_size, face = "bold"),
      plot.margin = margin(10, 10, 10, 10)
    )
}

cb_palette <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7",
                "#F0E442", "#56B4E9", "#E69F00", "#000000")

# --- Load and prepare data ---
data(lung, package = "survival")
lung <- lung %>%
  mutate(
    sex_label = factor(sex, levels = c(1, 2), labels = c("Male", "Female")),
    status_event = status - 1  # Convert to 0/1
  ) %>%
  filter(!is.na(ph.ecog) & !is.na(ph.karno) & !is.na(wt.loss))

# =============================================================================
# Panel A: Kaplan-Meier Curve by Sex
# =============================================================================
fit_km <- survfit(Surv(time, status_event) ~ sex_label, data = lung)

# Using ggsurvplot for KM curve
p_km <- ggsurvplot(
  fit_km,
  data = lung,
  palette = cb_palette[1:2],
  risk.table = FALSE,
  pval = TRUE,
  pval.coord = c(700, 0.9),
  conf.int = TRUE,
  conf.int.alpha = 0.15,
  xlab = "Time (days)",
  ylab = "Survival probability",
  legend.labs = c("Male", "Female"),
  legend.title = "Sex",
  ggtheme = theme_publication(),
  title = "A"
)

panel_a <- p_km$plot +
  theme(legend.position = c(0.8, 0.85),
        legend.background = element_rect(fill = "white", colour = "grey80"),
        plot.title = element_text(size = 14, face = "bold"))

# =============================================================================
# Panel B: Forest Plot of Hazard Ratios from Multivariable Cox Model
# =============================================================================
cox_model <- coxph(Surv(time, status_event) ~ age + sex_label + ph.ecog +
                     ph.karno + wt.loss,
                   data = lung)

# Extract coefficients
cox_tidy <- tidy(cox_model, conf.int = TRUE, exponentiate = TRUE)
cox_tidy <- cox_tidy %>%
  mutate(
    label = case_when(
      term == "age" ~ "Age (per year)",
      term == "sex_labelFemale" ~ "Female vs Male",
      term == "ph.ecog" ~ "ECOG PS (per level)",
      term == "ph.karno" ~ "Karnofsky (per 10 pts)",
      term == "wt.loss" ~ "Weight loss (per kg)"
    ),
    # Rescale Karnofsky to per-10-point increase for interpretability
    estimate = ifelse(term == "ph.karno", estimate^10, estimate),
    conf.low = ifelse(term == "ph.karno", conf.low^10, conf.low),
    conf.high = ifelse(term == "ph.karno", conf.high^10, conf.high)
  )

panel_b <- ggplot(cox_tidy, aes(x = estimate, y = reorder(label, estimate))) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 height = 0.2, colour = cb_palette[1], linewidth = 0.8) +
  geom_point(size = 3, colour = cb_palette[1]) +
  geom_text(aes(label = sprintf("%.2f (%.2f-%.2f)", estimate, conf.low, conf.high)),
            hjust = -0.1, size = 3, nudge_y = 0.25) +
  scale_x_log10() +
  labs(x = "Hazard Ratio (95% CI)",
       y = "",
       title = "B") +
  theme_publication() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 14, face = "bold")
  )

# =============================================================================
# Panel C: Calibration Plot for 1-Year Survival Prediction
# =============================================================================

# Predict 1-year (365 day) survival probabilities
# Use Cox model baseline hazard
surv_pred <- summary(survfit(cox_model), times = 365)

# For calibration, split into deciles of predicted risk
lung$pred_1yr <- 1 - predict(cox_model, type = "expected",
                              newdata = lung) / max(lung$time) * 365
# Alternative: use survfit-based prediction
baseline_surv <- basehaz(cox_model, centered = FALSE)
# Get linear predictor
lung$lp <- predict(cox_model, type = "lp")

# Simple calibration: bin by predicted risk deciles
lung$pred_surv <- exp(-predict(cox_model, type = "expected"))

# Create risk groups based on linear predictor
lung$risk_group <- cut(lung$lp, breaks = quantile(lung$lp, probs = seq(0, 1, 0.1)),
                       include.lowest = TRUE, labels = FALSE)

# Calculate observed 1-year survival in each group
cal_data <- lung %>%
  group_by(risk_group) %>%
  summarise(
    n = n(),
    events = sum(status_event),
    # KM estimate at 1 year
    obs_surv = {
      fit <- survfit(Surv(time, status_event) ~ 1, data = cur_data())
      s <- summary(fit, times = 365)
      if (length(s$surv) > 0) s$surv else NA
    },
    mean_lp = mean(lp),
    .groups = "drop"
  ) %>%
  filter(!is.na(obs_surv))

# Get predicted survival at each mean LP
h0_365 <- approx(baseline_surv$time, baseline_surv$hazard, xout = 365)$y
if (is.na(h0_365)) h0_365 <- max(baseline_surv$hazard)
cal_data$pred_surv <- exp(-h0_365 * exp(cal_data$mean_lp))

panel_c <- ggplot(cal_data, aes(x = 1 - pred_surv, y = 1 - obs_surv)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
  geom_point(size = 3, colour = cb_palette[1]) +
  geom_smooth(method = "loess", se = TRUE, colour = cb_palette[2],
              fill = cb_palette[2], alpha = 0.2) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Predicted 1-year mortality risk",
       y = "Observed 1-year mortality",
       title = "C") +
  theme_publication() +
  theme(plot.title = element_text(size = 14, face = "bold"))

# =============================================================================
# Combine panels using patchwork
# =============================================================================
# Panel A takes the top row, B and C share the bottom row
combined <- panel_a / (panel_b | panel_c) +
  plot_layout(heights = c(1, 1))

# Save at publication quality
# Double-column layout: 183 mm wide
ggsave("figure_survival_analysis.tiff",
       plot = combined,
       width = 183, height = 200,
       units = "mm", dpi = 300,
       compression = "lzw")

ggsave("figure_survival_analysis.pdf",
       plot = combined,
       width = 183, height = 200,
       units = "mm", device = cairo_pdf)

print(combined)

cat("\nMulti-panel figure saved as:\n")
cat("  figure_survival_analysis.tiff (300 DPI, TIFF)\n")
cat("  figure_survival_analysis.pdf (vector PDF)\n")
cat("\nSpecifications:\n")
cat("  Width: 183 mm (double-column)\n")
cat("  Resolution: 300 DPI\n")
cat("  Colour palette: colour-blind accessible\n")
cat("  Font: Arial\n")
