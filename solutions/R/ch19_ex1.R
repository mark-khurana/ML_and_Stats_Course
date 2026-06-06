# =============================================================================
# Chapter 19 - Exercise 1: Table 1 in R
# Using the lung dataset, create a publication-quality Table 1
# =============================================================================

library(tidyverse)
library(gtsummary)
library(gt)
library(survival)

# --- Load data ---
data(lung, package = "survival")

# Prepare data: recode sex for clarity
lung_clean <- lung %>%
  mutate(
    sex = factor(sex, levels = c(1, 2), labels = c("Male", "Female")),
    ph.ecog = factor(ph.ecog, levels = 0:4,
                     labels = c("Fully active", "Restricted",
                                "Ambulatory", "Limited self-care",
                                "Completely disabled")),
    status = factor(status, levels = c(1, 2), labels = c("Censored", "Dead"))
  )

# --- Part (a): Create Table 1 stratified by sex ---
table1 <- lung_clean %>%
  select(sex, age, ph.ecog, ph.karno, meal.cal, wt.loss) %>%
  tbl_summary(
    by = sex,
    statistic = list(
      all_continuous() ~ "{mean} ({sd})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = list(
      all_continuous() ~ 1,
      all_categorical() ~ c(0, 1)
    ),
    label = list(
      age ~ "Age, years",
      ph.ecog ~ "ECOG performance status",
      ph.karno ~ "Karnofsky performance score",
      meal.cal ~ "Meal calories, kcal/day",
      wt.loss ~ "Weight loss, kg (last 6 months)"
    ),
    missing_text = "Missing"
  ) %>%
  add_overall() %>%
  add_n()

# --- Part (b): Add SMDs instead of p-values ---
# gtsummary supports SMD via add_difference()
# For Table 1 with SMDs, we use a different approach
table1_smd <- lung_clean %>%
  select(sex, age, ph.ecog, ph.karno, meal.cal, wt.loss) %>%
  tbl_summary(
    by = sex,
    statistic = list(
      all_continuous() ~ "{mean} ({sd})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = list(
      all_continuous() ~ 1,
      all_categorical() ~ c(0, 1)
    ),
    label = list(
      age ~ "Age, years",
      ph.ecog ~ "ECOG performance status",
      ph.karno ~ "Karnofsky performance score",
      meal.cal ~ "Meal calories, kcal/day",
      wt.loss ~ "Weight loss, kg (last 6 months)"
    ),
    missing_text = "Missing"
  ) %>%
  add_overall() %>%
  add_n() %>%
  add_difference(
    estimate_fun = list(all_continuous() ~ function(x) style_sigfig(x, digits = 3)),
    pvalue_fun = ~ style_pvalue(.x, digits = 3)
  ) %>%
  modify_header(label = "**Characteristic**") %>%
  modify_spanning_header(c("stat_1", "stat_2") ~ "**Sex**") %>%
  bold_labels()

# Compute SMDs manually for continuous variables
compute_smd <- function(x, group) {
  g1 <- x[group == "Male"]
  g2 <- x[group == "Female"]
  g1 <- g1[!is.na(g1)]
  g2 <- g2[!is.na(g2)]
  pooled_sd <- sqrt((var(g1) + var(g2)) / 2)
  if (pooled_sd == 0) return(0)
  (mean(g1) - mean(g2)) / pooled_sd
}

cat("Standardised Mean Differences (Male vs Female):\n")
cat("  Age:", round(compute_smd(lung_clean$age, lung_clean$sex), 3), "\n")
cat("  Karnofsky:", round(compute_smd(lung_clean$ph.karno, lung_clean$sex), 3), "\n")
cat("  Meal calories:", round(compute_smd(lung_clean$meal.cal, lung_clean$sex), 3), "\n")
cat("  Weight loss:", round(compute_smd(lung_clean$wt.loss, lung_clean$sex), 3), "\n")

# Create the final publication table with gt formatting
final_table <- table1_smd %>%
  as_gt() %>%
  gt::tab_header(
    title = "Table 1. Baseline Characteristics of the Study Population",
    subtitle = "NCCTG Lung Cancer Dataset, Stratified by Sex"
  ) %>%
  gt::tab_footnote(
    footnote = "Values are mean (SD) for continuous variables and n (%) for categorical variables. SMD = standardised mean difference.",
    locations = gt::cells_title()
  )

print(final_table)

# --- Part (c): Export to Word and LaTeX ---
# Export to Word (.docx)
# gt::gtsave(final_table, "table1_lung.docx")
cat("\nTo export to Word: gt::gtsave(final_table, 'table1_lung.docx')\n")

# Export to LaTeX
# gt::as_latex(final_table) %>% writeLines("table1_lung.tex")
cat("To export to LaTeX: gt::as_latex(final_table)\n")

# Export to HTML
# gt::gtsave(final_table, "table1_lung.html")
cat("To export to HTML: gt::gtsave(final_table, 'table1_lung.html')\n")

cat("\nNote: SMDs < 0.1 indicate good balance. In an RCT, p-values for\n")
cat("baseline comparisons are inappropriate because any differences are\n")
cat("due to chance. In observational studies, SMDs are preferred over\n")
cat("p-values because they are independent of sample size.\n")
