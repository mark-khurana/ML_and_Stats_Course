# =============================================================================
# Chapter 17 - Exercise 6: Target trial emulation and immortal time bias
# Early vs delayed metformin initiation and 5-year cardiovascular events
# =============================================================================

# Libraries -------------------------------------------------------------------
# pip install numpy pandas lifelines
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# =============================================================================
# (a) The target trial protocol
# =============================================================================
# Write this table BEFORE touching the data. Every ambiguity you leave here is a
# place where bias can enter later.
#
#  Component            | Target trial we wish we could run          | How we emulate it in the EHR
#  ---------------------|-------------------------------------------|------------------------------------------
#  Eligibility          | Adults 40-75, newly diagnosed T2DM, no    | Same criteria at the diagnosis date;
#                       | prior CVD, no prior metformin, alive and  | require >=12 months of prior database
#                       | event-free at time zero                   | coverage so "new" really means new
#  Treatment strategies | (1) initiate metformin within 3 months    | Same two strategies, from dispensing
#                       | (2) initiate between 3 and 12 months      | records in the corresponding windows
#  Assignment           | Randomised                                | NOT random. Assume conditional
#                       |                                           | exchangeability given measured
#                       |                                           | covariates; adjust by IPW/g-computation
#  Time zero            | Date of randomisation                     | End of the 3-month window (the
#                       |                                           | landmark), among patients alive and
#                       |                                           | event-free at that point
#  Outcome              | First MACE within 5 years of time zero    | Same, from linked hospital/death records
#  Causal contrast      | Intention-to-treat (per-protocol as       | ITT analogue = strategies as assigned at
#                       | secondary)                                | time zero; per-protocol needs
#                       |                                           | clone-censor-weight
#  Analysis plan        | Cox model / cumulative incidence by       | Same, weighted; report absolute risks
#                       | assigned strategy                         | and not only hazard ratios
#
# The single most important row is TIME ZERO. Get it wrong and no amount of
# covariate adjustment will rescue the analysis, as parts (b) and (d) show.

# =============================================================================
# The data: metformin timing genuinely does not matter
# =============================================================================
rng = np.random.default_rng(42)
n = 20_000
MAXFU = 5       # 5 years of follow-up
WINDOW = 0.25   # the 3-month grace period, in years

# Time to a first cardiovascular event, generated with NO reference at all to
# when (or whether) the patient starts metformin. The true effect of early versus
# late initiation is therefore EXACTLY zero: hazard ratio 1.00.
# A rate of 0.06/year gives roughly a 26% five-year MACE risk, in the right
# ballpark for a newly diagnosed T2DM cohort.
event_time = rng.exponential(1 / 0.06, n)

# When this patient WOULD start metformin, if they live long enough to do so.
init_time = rng.exponential(1 / 1.5, n)

obs_time = np.minimum(event_time, MAXFU)
had_event = (event_time <= MAXFU).astype(int)

print(f"Cohort: {n} patients, {100 * had_event.mean():.1f}% had a MACE within "
      "5 years")
print("TRUE hazard ratio, by construction: 1.00 (timing has no effect at all)")


def cox_hr(duration, event, exposure, covariates=None):
    """Fit a Cox model and return (HR, lo, hi) for the exposure."""
    data = pd.DataFrame({"t": np.asarray(duration, float),
                         "e": np.asarray(event, int),
                         "x": np.asarray(exposure, int)})
    if covariates is not None:
        for name, values in covariates.items():
            data[name] = np.asarray(values, float)
    fit = CoxPHFitter().fit(data, duration_col="t", event_col="e")
    hr = float(np.exp(fit.params_["x"]))
    lo, hi = np.exp(fit.confidence_intervals_.loc["x"].to_numpy())
    return hr, float(lo), float(hi)


def report(label, result):
    hr, lo, hi = result
    print(f"{label:<40} HR = {hr:.2f}  (95% CI {lo:.2f}, {hi:.2f})")


# =============================================================================
# (b) The naive analysis: a 3-month window, but the clock still runs from
#     diagnosis
# =============================================================================
# "Early initiator" = started inside the 3-month window. Note the second
# condition: you cannot collect a prescription after your event. THAT is where
# the bias comes from.
early = (init_time <= WINDOW) & (init_time < event_time)

print("\n=== (b) Naive: early vs late initiator, clock from diagnosis ===")
report("Early vs late initiator", cox_hr(obs_time, had_event, early))

n_events_in_window = int((event_time <= WINDOW).sum())
n_pending = int(((event_time <= WINDOW) & (init_time <= WINDOW)
                 & (init_time >= event_time)).sum())
print(f"\n{n_events_in_window} patients had their event inside the 3-month "
      "window. Every one of them")
print(f"is classified as a LATE initiator -- including {n_pending} who were on "
      "course to")
print("start metformin early and simply did not get the chance.")
print("""
So membership of the 'early' group requires surviving the first 3 months, and we
then count those 3 months as follow-up in which no early initiator could possibly
have had an event. That stretch of guaranteed survival is the IMMORTAL TIME.

At 3 months the resulting bias is small, and it is worth being honest about why:
3 months of immortal time is little next to 5 years of follow-up. Which leads
directly to the useful diagnostic below.""")

# =============================================================================
# (b, continued) How big is the bias? It scales with the exposure window
# =============================================================================
print("\n=== (b) The bias grows with the length of the exposure window ===")

scan = []
for w in [0.25, 0.5, 1, 2]:
    is_early = (init_time <= w) & (init_time < event_time)
    naive_hr, *_ = cox_hr(obs_time, had_event, is_early)
    at_risk_w = event_time > w
    land_hr, *_ = cox_hr(np.minimum(event_time[at_risk_w], MAXFU) - w,
                         (event_time[at_risk_w] <= MAXFU).astype(int),
                         is_early[at_risk_w])
    scan.append({"window_months": w * 12, "naive_HR": naive_hr,
                 "landmark_HR": land_hr,
                 "excluded_by_landmark": int((~at_risk_w).sum())})
print(pd.DataFrame(scan).round(3).to_string(index=False))

print("""
The naive hazard ratio drifts further from the truth the longer the window:
barely biased at 3 months, clearly biased at 12, and absurd at 24, where a drug
that does nothing appears to cut cardiovascular events by roughly three quarters.
The landmark column stays close to 1.00 throughout.

The published immortal-time-bias disasters are the extreme case of this table:
they defined exposure as 'ever dispensed the drug during follow-up', which is a
window as long as the study itself. That is why the effect sizes in those papers
were not merely optimistic but implausible.

So when you read someone else's observational drug study, the first question is:
how long was the exposure-definition window, relative to the follow-up? If the
answer is 'the whole study', stop reading.""")

# =============================================================================
# (c) Fix it with a landmark design
# =============================================================================
# Two changes, both about time zero:
#   1. include only patients still alive and event-free at the end of the window;
#   2. start the clock AT the end of the window rather than at diagnosis.
print("\n=== (c) The emulated target trial (landmark at 3 months) ===")

at_risk = event_time > WINDOW
emulated = cox_hr(np.minimum(event_time[at_risk], MAXFU) - WINDOW,
                  (event_time[at_risk] <= MAXFU).astype(int),
                  early[at_risk])
report("Emulated trial, clock from landmark", emulated)

print(f"\n{int((~at_risk).sum())} of {n} patients are excluded because their "
      "event happened before")
print("time zero. They leave BOTH arms, which is the point -- a real trial could")
print("not have enrolled them either. Now nobody's group membership depends on")
print("surviving any of the time we go on to analyse.")

print("\nWhich of the two changes is doing the work? Worth checking rather than")
print("assuming, and here we use the 24-month window where the bias is large:")

W_BIG = 2.0
early_big = (init_time <= W_BIG) & (init_time < event_time)
at_risk_big = event_time > W_BIG

report("  24-mo window, no fix at all",
       cox_hr(obs_time, had_event, early_big))
report("  exclusion only, clock from diagnosis",
       cox_hr(np.minimum(event_time[at_risk_big], MAXFU),
              (event_time[at_risk_big] <= MAXFU).astype(int),
              early_big[at_risk_big]))
report("  exclusion + clock moved (full fix)",
       cox_hr(np.minimum(event_time[at_risk_big], MAXFU) - W_BIG,
              (event_time[at_risk_big] <= MAXFU).astype(int),
              early_big[at_risk_big]))

print("""
The EXCLUSION does almost all of the work, because shifting every patient's clock
by the same constant does not change the order in which events occur, and a Cox
model only uses that order. Moving time zero starts to matter as soon as entry is
staggered, follow-up is administratively censored at a calendar date, or you want
absolute risks rather than a hazard ratio -- all of which are true of real data.
Do both.""")

# =============================================================================
# (d) Why adjusting for more covariates would not have helped
# =============================================================================
print("\n=== (d) Why covariate adjustment cannot fix (b) ===")

# Add a genuinely prognostic covariate -- exactly the kind of variable a reviewer
# would demand you adjust for -- and adjust for it thoroughly. We use the
# 24-month window so the bias is large enough to see clearly.
rng2 = np.random.default_rng(7)
frailty = rng2.normal(0, 1, n)
event_time2 = rng2.exponential(1 / (0.06 * np.exp(0.5 * frailty)))
init_time2 = rng2.exponential(1 / 1.5, n)
obs2 = np.minimum(event_time2, MAXFU)
ev2 = (event_time2 <= MAXFU).astype(int)
early2 = (init_time2 <= W_BIG) & (init_time2 < event_time2)
at_risk2 = event_time2 > W_BIG

report("Naive (24-mo window), unadjusted", cox_hr(obs2, ev2, early2))
report("Naive (24-mo window), + frailty",
       cox_hr(obs2, ev2, early2, {"frailty": frailty}))
report("Emulated trial, + frailty",
       cox_hr(np.minimum(event_time2[at_risk2], MAXFU) - W_BIG,
              (event_time2[at_risk2] <= MAXFU).astype(int),
              early2[at_risk2], {"frailty": frailty[at_risk2]}))

print("""
Adjustment barely moves the biased estimate, and here is why. Covariate
adjustment addresses CONFOUNDING: treated and untreated patients differing in
ways that also affect the outcome. Immortal time bias is not confounding. It is a
bookkeeping error about TIME -- we have credited one group with follow-up during
which it was impossible for them to have had an event. No covariate in the
dataset encodes that, so no covariate can correct for it. Note that the
simulation in (b) contained NO confounders whatsoever, and the bias was still
there.

The general lesson: some biases are design problems, and design problems need
design solutions. The target trial framework earns its keep precisely because it
forces the design decisions -- eligibility, assignment, and time zero -- to be
made explicitly, and before the analysis.

--- The grace period, and when the landmark is not enough ---
A patient who starts metformin in month 3 spent months 1-2 untreated while
counted in the 'early' arm. The landmark design tolerates that because it is an
INTENTION-TO-TREAT analogue: we compare strategies as assigned, not treatments as
received, exactly as a trial's ITT analysis does.

If you want the per-protocol effect, you need CLONE-CENSOR-WEIGHT:
  1. create a copy ('clone') of each eligible patient in each strategy arm;
  2. censor each clone at the moment its actual behaviour departs from its
     assigned strategy;
  3. re-weight by the inverse probability of remaining uncensored, to correct for
     the fact that departing is not random.
Step 3 is inverse probability weighting again, doing the same job for informative
censoring that it does for confounding earlier in the chapter.

And note the honest cost of the landmark: it discards every event in the first
3 months. If a treatment acts fastest early on, you will understate it. That is a
real trade-off to state in the paper, not a reason to go back to timing from
diagnosis.""")
