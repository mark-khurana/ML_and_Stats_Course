# =============================================================================
# Chapter 17 - Exercise 6: Target trial emulation and immortal time bias
# Early vs delayed metformin initiation and 5-year cardiovascular events
# =============================================================================
#
# Libraries -------------------------------------------------------------------
library(survival) # Surv(), coxph() -- MUST be loaded or coxph() is not found

# Print a Cox model's exposure hazard ratio. Takes only the FIRST coefficient,
# so it works for adjusted models too.
report <- function(label, fit) {
  hr <- exp(coef(fit))[1]
  ci <- exp(confint(fit))
  ci <- if (is.matrix(ci)) ci[1, ] else ci
  cat(sprintf("%-42s HR = %.2f  (95%% CI %.2f, %.2f)\n", label, hr, ci[1], ci[2]))
}

# =============================================================================
# (a) The target trial protocol
# =============================================================================
# Write this table BEFORE touching the data. Every ambiguity you leave here is
# a place where bias can enter later.
#
#  Component            | Target trial we wish we could run          | How we emulate it in the EHR
#  ---------------------|-------------------------------------------|-------------------------------------------
#  Eligibility          | Adults 40-75, newly diagnosed T2DM, no    | Same criteria at the diagnosis date;
#                       | prior CVD, no prior metformin, alive and  | require >=12 months of prior database
#                       | event-free at time zero                   | coverage so "new" really means new
#  Treatment strategies | (1) initiate metformin within 3 months    | Same two strategies, defined from
#                       | (2) initiate between 3 and 12 months      | dispensing records in those windows
#  Assignment           | Randomised                                | NOT random. Assume conditional
#                       |                                           | exchangeability given measured
#                       |                                           | covariates; adjust by IPW or g-computation
#  Time zero            | Date of randomisation                     | End of the 3-month window (the landmark),
#                       |                                           | among patients alive and event-free then
#  Outcome              | First MACE within 5 years of time zero    | Same, from linked hospital/death records
#  Causal contrast      | Intention-to-treat (per-protocol as       | ITT analogue = compare strategies as
#                       | secondary)                                | assigned at time zero; per-protocol needs
#                       |                                           | clone-censor-weight
#  Analysis plan        | Cox model / cumulative incidence by       | Same, weighted; report absolute risks and
#                       | assigned strategy                         | not only hazard ratios
#
# The single most important row is TIME ZERO. Get it wrong and no amount of
# covariate adjustment will rescue the analysis, as parts (b) and (d) show.

# =============================================================================
# The data: metformin timing genuinely does not matter
# =============================================================================
set.seed(42)
n <- 20000
MAXFU <- 5      # 5 years of follow-up
WINDOW <- 0.25  # the 3-month grace period, in years

# Time to a first cardiovascular event, generated with NO reference at all to
# when (or whether) the patient starts metformin. The true effect of early
# versus late initiation is therefore EXACTLY zero: hazard ratio 1.00.
# A rate of 0.06/year gives roughly a 26% five-year MACE risk, in the right
# ballpark for a newly diagnosed T2DM cohort.
event_time <- rexp(n, rate = 0.06)

# When this patient WOULD start metformin, if they live long enough to do so.
init_time <- rexp(n, rate = 1.5)

obs_time <- pmin(event_time, MAXFU)
had_event <- as.integer(event_time <= MAXFU)

cat(sprintf("Cohort: %d patients, %.1f%% had a MACE within 5 years\n",
            n, 100 * mean(had_event)))
cat("TRUE hazard ratio, by construction: 1.00 (timing has no effect at all)\n")

# =============================================================================
# (b) The naive analysis: a 3-month window, but the clock still runs from
#     diagnosis
# =============================================================================
# "Early initiator" = started inside the 3-month window. Note the second
# condition: you cannot collect a prescription after your event. THAT is where
# the bias comes from.
early <- init_time <= WINDOW & init_time < event_time

cat("\n=== (b) Naive: early vs late initiator, clock from diagnosis ===\n")
report("Early vs late initiator", coxph(Surv(obs_time, had_event) ~ early))

n_events_in_window <- sum(event_time <= WINDOW)
n_pending <- sum(event_time <= WINDOW & init_time <= WINDOW &
  init_time >= event_time)
cat(sprintf(
  "\n%d patients had their event inside the 3-month window. Every one of them\n",
  n_events_in_window
))
cat(sprintf(
  "is classified as a LATE initiator -- including %d who were on course to\n",
  n_pending
))
cat("start metformin early and simply did not get the chance.\n")
cat("\nSo membership of the 'early' group requires surviving the first 3 months,\n")
cat("and we then count those 3 months as follow-up in which no early initiator\n")
cat("could possibly have had an event. That stretch of guaranteed survival is\n")
cat("the IMMORTAL TIME.\n")
cat("\nAt 3 months the resulting bias is small, and it is worth being honest\n")
cat("about why: 3 months of immortal time is little next to 5 years of\n")
cat("follow-up. Which leads directly to the useful diagnostic below.\n")

# =============================================================================
# (b, continued) How big is the bias? It scales with the exposure window
# =============================================================================
cat("\n=== (b) The bias grows with the length of the exposure window ===\n")

window_scan <- t(vapply(c(0.25, 0.5, 1, 2), function(w) {
  is_early <- init_time <= w & init_time < event_time
  naive_fit <- coxph(Surv(obs_time, had_event) ~ is_early)
  at_risk_w <- event_time > w
  land_fit <- coxph(
    Surv(pmin(event_time[at_risk_w], MAXFU) - w,
         as.integer(event_time[at_risk_w] <= MAXFU)) ~ is_early[at_risk_w]
  )
  c(
    window_months = w * 12,
    naive_HR = unname(exp(coef(naive_fit))[1]),
    landmark_HR = unname(exp(coef(land_fit))[1]),
    excluded_by_landmark = sum(!at_risk_w)
  )
}, numeric(4)))

print(round(as.data.frame(window_scan), 3), row.names = FALSE)

cat("\nThe naive hazard ratio drifts further from the truth the longer the\n")
cat("window: barely biased at 3 months, clearly biased at 12, and absurd at 24,\n")
cat("where a drug that does nothing appears to cut cardiovascular events by\n")
cat("roughly three quarters. The landmark column stays close to 1.00 throughout.\n")
cat("\nThe published immortal-time-bias disasters are the extreme case of this\n")
cat("table: they defined exposure as 'ever dispensed the drug during follow-up',\n")
cat("which is a window as long as the study itself. That is why the effect sizes\n")
cat("in those papers were not merely optimistic but implausible.\n")
cat("\nSo when you read someone else's observational drug study, the first\n")
cat("question is: how long was the exposure-definition window, relative to the\n")
cat("follow-up? If the answer is 'the whole study', stop reading.\n")

# =============================================================================
# (c) Fix it with a landmark design
# =============================================================================
# Two changes, both about time zero:
#   1. include only patients still alive and event-free at the end of the window;
#   2. start the clock AT the end of the window rather than at diagnosis.
cat("\n=== (c) The emulated target trial (landmark at 3 months) ===\n")

at_risk <- event_time > WINDOW
emulated <- coxph(
  Surv(pmin(event_time[at_risk], MAXFU) - WINDOW,
       as.integer(event_time[at_risk] <= MAXFU)) ~ early[at_risk]
)
report("Emulated trial, clock from landmark", emulated)

cat(sprintf(
  "\n%d of %d patients are excluded because their event happened before time\n",
  sum(!at_risk), n
))
cat("zero. They leave BOTH arms, which is the point -- a real trial could not\n")
cat("have enrolled them either. Now nobody's group membership depends on\n")
cat("surviving any of the time we go on to analyse.\n")

cat("\nWhich of the two changes is doing the work? Worth checking rather than\n")
cat("assuming, and here we use the 24-month window where the bias is large:\n")
W_BIG <- 2
early_big <- init_time <= W_BIG & init_time < event_time
at_risk_big <- event_time > W_BIG

report("  24-mo window, no fix at all", coxph(
  Surv(obs_time, had_event) ~ early_big
))
report("  exclusion only, clock from diagnosis", coxph(
  Surv(pmin(event_time[at_risk_big], MAXFU),
       as.integer(event_time[at_risk_big] <= MAXFU)) ~ early_big[at_risk_big]
))
report("  exclusion + clock moved (full fix)", coxph(
  Surv(pmin(event_time[at_risk_big], MAXFU) - W_BIG,
       as.integer(event_time[at_risk_big] <= MAXFU)) ~ early_big[at_risk_big]
))

cat("\nThe EXCLUSION does almost all of the work, because shifting every\n")
cat("patient's clock by the same constant does not change the order in which\n")
cat("events occur, and a Cox model only uses that order. Moving time zero starts\n")
cat("to matter as soon as entry is staggered, follow-up is administratively\n")
cat("censored at a calendar date, or you want absolute risks rather than a\n")
cat("hazard ratio -- all of which are true of real data. Do both.\n")

# =============================================================================
# (d) Why adjusting for more covariates would not have helped
# =============================================================================
cat("\n=== (d) Why covariate adjustment cannot fix (b) ===\n")

# Add a genuinely prognostic covariate -- exactly the kind of variable a
# reviewer would demand you adjust for -- and adjust for it thoroughly. We use
# the 24-month window so the bias is large enough to see clearly.
set.seed(7)
frailty <- rnorm(n)
event_time2 <- rexp(n, rate = 0.06 * exp(0.5 * frailty))
init_time2 <- rexp(n, rate = 1.5)
obs2 <- pmin(event_time2, MAXFU)
ev2 <- as.integer(event_time2 <= MAXFU)
early2 <- init_time2 <= W_BIG & init_time2 < event_time2
at_risk2 <- event_time2 > W_BIG

report("Naive (24-mo window), unadjusted",
       coxph(Surv(obs2, ev2) ~ early2))
report("Naive (24-mo window), + frailty",
       coxph(Surv(obs2, ev2) ~ early2 + frailty))
report("Emulated trial, + frailty", coxph(
  Surv(pmin(event_time2[at_risk2], MAXFU) - W_BIG,
       as.integer(event_time2[at_risk2] <= MAXFU))
  ~ early2[at_risk2] + frailty[at_risk2]
))

cat("\nAdjustment barely moves the biased estimate, and here is why. Covariate\n")
cat("adjustment addresses CONFOUNDING: treated and untreated patients differing\n")
cat("in ways that also affect the outcome. Immortal time bias is not\n")
cat("confounding. It is a bookkeeping error about TIME -- we have credited one\n")
cat("group with follow-up during which it was impossible for them to have had\n")
cat("an event. No covariate in the dataset encodes that, so no covariate can\n")
cat("correct for it. Note that the simulation in (b) contained NO confounders\n")
cat("whatsoever, and the bias was still there.\n")
cat("\nThe general lesson: some biases are design problems, and design problems\n")
cat("need design solutions. The target trial framework earns its keep precisely\n")
cat("because it forces the design decisions -- eligibility, assignment, and\n")
cat("time zero -- to be made explicitly, and before the analysis.\n")

# =============================================================================
# One remaining wrinkle: the grace period
# =============================================================================
cat("\n--- The grace period, and when the landmark is not enough ---\n")
cat("A patient who starts metformin in month 3 spent months 1-2 untreated while\n")
cat("counted in the 'early' arm. The landmark design tolerates that because it\n")
cat("is an INTENTION-TO-TREAT analogue: we compare strategies as assigned, not\n")
cat("treatments as received, exactly as a trial's ITT analysis does.\n")
cat("\nIf you want the per-protocol effect, you need CLONE-CENSOR-WEIGHT:\n")
cat("  1. create a copy ('clone') of each eligible patient in each strategy arm;\n")
cat("  2. censor each clone at the moment its actual behaviour departs from its\n")
cat("     assigned strategy;\n")
cat("  3. re-weight by the inverse probability of remaining uncensored, to\n")
cat("     correct for the fact that departing is not random.\n")
cat("Step 3 is inverse probability weighting again, doing the same job for\n")
cat("informative censoring that it does for confounding earlier in the chapter.\n")
cat("\nAnd note the honest cost of the landmark: it discards every event in the\n")
cat("first 3 months. If a treatment acts fastest early on, you will understate\n")
cat("it. That is a real trade-off to state in the paper, not a reason to go back\n")
cat("to timing from diagnosis.\n")
