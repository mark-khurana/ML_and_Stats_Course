# =============================================================================
# Chapter 17 - Exercise 1: Build a DAG and derive the adjustment set
# ACE inhibitor use and acute kidney injury (AKI) in hospitalised patients
# =============================================================================

# Libraries -------------------------------------------------------------------
# pip install networkx statsmodels
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -----------------------------------------------------------------------------
# (a) Relevant variables, and the causal ROLE of each
# -----------------------------------------------------------------------------
# The role matters more than the list: it decides whether you adjust or not.
#
# CONFOUNDERS (arrow into BOTH ACEi and AKI) -> MUST adjust
#   1. Baseline kidney function (eGFR / creatinine): a reason to prescribe an
#      ACE inhibitor (renoprotection) AND an independent risk factor for AKI.
#   2. Heart failure: a major indication for ACE inhibitors AND independently
#      raises AKI risk through haemodynamic changes.
#   3. Age: older patients are more likely to be treated and more likely to
#      develop AKI.
#   4. Diabetes / hypertension: both are indications for ACE inhibitors and
#      both raise AKI risk.
#
# MEDIATOR (on the causal path) -> do NOT adjust for a total effect
#   5. Renal perfusion pressure: part of HOW an ACE inhibitor precipitates AKI.
#      Adjust for it and you remove part of the effect you want to measure.
#
# COMPETING CAUSE (arrow into AKI only) -> optional; harmless but unnecessary
#   6. Concomitant nephrotoxic drugs (NSAIDs, contrast).
#
# COLLIDER (arrow in from BOTH) -> NEVER adjust
#   7. ICU admission: caused by ACEi-related complications AND by AKI itself.

# -----------------------------------------------------------------------------
# (b) Encode the DAG
# -----------------------------------------------------------------------------
edges = [
    ("Age", "ACEi"), ("Age", "AKI"), ("Age", "BaselineEGFR"),
    ("BaselineEGFR", "ACEi"), ("BaselineEGFR", "AKI"),
    ("HeartFailure", "ACEi"), ("HeartFailure", "AKI"),
    ("Diabetes", "ACEi"), ("Diabetes", "AKI"), ("Diabetes", "BaselineEGFR"),
    ("Hypertension", "ACEi"), ("Hypertension", "AKI"),
    ("NephrotoxicDrugs", "AKI"),
    ("ACEi", "AKI"),
    ("ACEi", "RenalPerfusion"), ("RenalPerfusion", "AKI"),
    ("ACEi", "ICUAdmission"), ("AKI", "ICUAdmission"),
]
dag = nx.DiGraph(edges)
assert nx.is_directed_acyclic_graph(dag), "a DAG must not contain cycles"

EXPOSURE, OUTCOME = "ACEi", "AKI"

# -----------------------------------------------------------------------------
# (c) Check an adjustment set with the backdoor criterion
# -----------------------------------------------------------------------------
# Python has no direct equivalent of R's dagitty::adjustmentSets(), but the
# backdoor criterion is short to implement with networkx:
#   1. delete every edge LEAVING the exposure (this removes the causal paths,
#      leaving only the backdoor paths behind);
#   2. the set Z is sufficient if, in that modified graph, exposure and outcome
#      are d-separated given Z;
#   3. and Z must not contain any descendant of the exposure.


def satisfies_backdoor(graph, exposure, outcome, adjust_for):
    """True if `adjust_for` blocks every backdoor path from exposure to outcome."""
    adjust_for = set(adjust_for)
    descendants = nx.descendants(graph, exposure)
    offenders = sorted(adjust_for & descendants)
    if offenders:
        return False, offenders
    backdoor_graph = graph.copy()
    backdoor_graph.remove_edges_from(list(graph.out_edges(exposure)))
    ok = nx.is_d_separator(backdoor_graph, {exposure}, {outcome}, adjust_for)
    return ok, []


confounders = {"Age", "BaselineEGFR", "HeartFailure", "Diabetes", "Hypertension"}

candidates = {
    "nothing (unadjusted)": set(),
    "the five confounders": confounders,
    "confounders + RenalPerfusion (a mediator)": confounders | {"RenalPerfusion"},
    "confounders + ICUAdmission (a collider)": confounders | {"ICUAdmission"},
    "confounders + NephrotoxicDrugs": confounders | {"NephrotoxicDrugs"},
}

print("--- (c) Which adjustment sets satisfy the backdoor criterion? ---")
for label, z in candidates.items():
    ok, offenders = satisfies_backdoor(dag, EXPOSURE, OUTCOME, z)
    verdict = "VALID  " if ok else "INVALID"
    note = f"  (contains descendants of {EXPOSURE}: {offenders})" if offenders else ""
    print(f"  {verdict}  adjust for {label}{note}")

# Find the minimal sufficient set by search rather than by intuition. We look
# for a d-separator in the backdoor graph, restricted to non-descendants of the
# exposure (so mediators and colliders below it can never be chosen).
backdoor_graph = dag.copy()
backdoor_graph.remove_edges_from(list(dag.out_edges(EXPOSURE)))
allowed = set(dag.nodes) - {EXPOSURE, OUTCOME} - nx.descendants(dag, EXPOSURE)
minimal = nx.find_minimal_d_separator(
    backdoor_graph, {EXPOSURE}, {OUTCOME}, restricted=allowed
)
print(f"\nMinimal sufficient adjustment set: {sorted(minimal)}")
print("Note what is EXCLUDED: RenalPerfusion (mediator), ICUAdmission")
print("(collider), NephrotoxicDrugs (opens no backdoor path).")

# -----------------------------------------------------------------------------
# (d) The collider, demonstrated on simulated data
# -----------------------------------------------------------------------------
# Simulate a small version of the DAG in which the ACE inhibitor has NO effect
# whatsoever on AKI (true log-odds = 0), then estimate the association
# three ways.

rng = np.random.default_rng(42)
n = 20_000


def expit(x):
    return 1 / (1 + np.exp(-x))


ckd = rng.binomial(1, 0.35, n)                               # one confounder
acei = rng.binomial(1, expit(-0.5 + 1.2 * ckd))              # prescribed more in CKD
aki = rng.binomial(1, expit(-2.0 + 1.5 * ckd + 0.0 * acei))  # TRUE effect = 0
icu = rng.binomial(1, expit(-2.0 + 1.0 * acei + 2.0 * aki))  # the collider

dat = pd.DataFrame(dict(ckd=ckd, acei=acei, aki=aki, icu=icu))


def log_odds(formula, data):
    fit = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(disp=0)
    return fit.params["acei"]


print("\n--- (d) Log-odds of ACEi on AKI (TRUE value = 0.000) ---")
print(f"Unadjusted (confounded by CKD)     : "
      f"{log_odds('aki ~ acei', dat):+.3f}")
print(f"Adjusted for the confounder (CKD)  : "
      f"{log_odds('aki ~ acei + ckd', dat):+.3f}   <- correct")
print(f"ALSO adjusted for ICU (a collider) : "
      f"{log_odds('aki ~ acei + ckd + icu', dat):+.3f}   <- bias re-introduced")
print(f"Restricted to ICU patients only    : "
      f"{log_odds('aki ~ acei + ckd', dat[dat.icu == 1]):+.3f}   <- same bias")

print("""
Interpretation:
Adjusting for CKD removes the confounding and recovers the truth (0).
Adding ICU admission -- or studying only ICU patients -- pushes the estimate
NEGATIVE, inventing a protective effect for a drug that does nothing. Why:
among ICU patients, someone who is NOT on an ACE inhibitor probably got there
because of their AKI, so 'no ACEi' starts to predict AKI. The association is
real inside the ICU and absent outside it.

Practical lesson: never adjust for, or select on, a variable that is a
consequence of both the exposure and the outcome.
""")
