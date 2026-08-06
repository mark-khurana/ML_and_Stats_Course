# Chapter 11, Exercise 3: Decision Curve Interpretation
# Conceptual exercise about a preeclampsia prediction model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import expit

# Simulate a plausible preeclampsia model to generate a decision curve
np.random.seed(42)
n = 2000

age = np.round(np.random.normal(30, 5, n))
bmi = np.round(np.random.normal(27, 5, n), 1)
nulliparous = np.random.binomial(1, 0.4, n)
history_pe = np.random.binomial(1, 0.05, n)
map_val = np.round(np.random.normal(85, 10, n))

lp = (-5.5 + 0.03 * age + 0.04 * bmi + 0.5 * nulliparous +
      1.5 * history_pe + 0.02 * map_val)
pe = np.random.binomial(1, expit(lp))
print(f"Preeclampsia rate: {pe.mean():.3f}")

X = np.column_stack([age, bmi, nulliparous, history_pe, map_val])
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X, pe)
pred_prob = model.predict_proba(X)[:, 1]
print(f"AUC: {roc_auc_score(pe, pred_prob):.3f}")

# ---- Generate decision curve ----
thresholds = np.arange(0.01, 0.51, 0.01)
n_total = len(pe)


def net_benefit(y_true, y_pred, threshold):
    pred_pos = (y_pred >= threshold).astype(int)
    tp = np.sum((pred_pos == 1) & (y_true == 1))
    fp = np.sum((pred_pos == 1) & (y_true == 0))
    return tp / n_total - fp / n_total * (threshold / (1 - threshold))


nb_model = [net_benefit(pe, pred_prob, t) for t in thresholds]
nb_all = [pe.mean() - (1 - pe.mean()) * t / (1 - t) for t in thresholds]

plt.figure(figsize=(9, 6))
plt.plot(thresholds, nb_model, color="steelblue", lw=2, label="Prediction Model")
plt.plot(thresholds, nb_all, color="coral", lw=2, ls="--", label="Treat All")
plt.axhline(y=0, color="black", lw=1, label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve: Preeclampsia Prediction Model (AUC ~ 0.82)")
plt.legend()
plt.ylim(-0.05, max(max(nb_model), max(nb_all)) * 1.1)
plt.tight_layout()
plt.savefig("ch11_ex3_decision_curve.png", dpi=150)
plt.show()

# ---- Part (a): Range of useful thresholds ----
print("\n=== Part (a): Range of Useful Thresholds ===")
print("The model is useful at thresholds where its net benefit curve")
print("exceeds BOTH 'treat all' and 'treat none'.")
print("From the decision curve, the model is useful approximately")
print("from about 2% to 30-40% threshold probability.")
print("Below ~2%, 'treat all' has similar or higher net benefit.")
print("Above ~30-40%, the model's net benefit approaches zero.")

# ---- Part (b): Counter-argument to 'AUC is only 0.82' ----
print("\n=== Part (b): Counter-argument ===")
print("An AUC of 0.82 is strong discrimination, not 'only'.")
print("More importantly, AUC does not directly measure clinical utility.")
print("")
print("The decision curve shows the model provides positive net benefit")
print("above both default strategies across the clinically relevant")
print("threshold range (e.g., 5-20%). This means using the model to")
print("guide monitoring decisions identifies more true preeclampsia")
print("cases per 'unnecessary' monitoring episode than either monitoring")
print("everyone or no one. Clinical utility -- not AUC -- determines")
print("whether a model should be used in practice.")

# ---- Part (c): Effect of action cost ----
print("\n=== Part (c): Effect of Action Cost ===")
print("The threshold probability reflects the cost-benefit ratio of acting.")
print("")
print("LOW-COST action (closer monitoring, extra visits):")
print("  - Clinicians use a LOW threshold (e.g., 3-5%)")
print("  - Many false positives are tolerable")
print("  - The relevant region of the decision curve is on the LEFT")
print("  - 'Treat all' remains competitive at low thresholds")
print("")
print("HIGH-COST action (prophylactic medication with side effects):")
print("  - Clinicians use a HIGHER threshold (e.g., 15-25%)")
print("  - More certainty is needed before acting")
print("  - The relevant region shifts RIGHT")
print("  - The model adds more value by avoiding unnecessary treatment")
print("")
print("The curve itself does not change, but the clinically relevant")
print("region changes depending on the cost of the follow-up action.")
