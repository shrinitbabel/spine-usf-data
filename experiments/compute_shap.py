"""
Compute SHAP values for the trained RSF on the full 546-patient cohort
(model-agnostic PermutationExplainer because sksurv's RSF is not natively
supported by shap.TreeExplainer).

Outputs:
  experiments/shap_values_cohort.npz   — SHAP values for all 546 patients
  figures/fig14_shap_summary.png       — population beeswarm summary
  figures/fig15_shap_global_bar.png    — global mean |SHAP| importance bar
"""

from __future__ import annotations
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
})

# Load
with open(ROOT / "models" / "rsf_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)
rsf = bundle["model"]
X_vals = np.asarray(bundle["X_vals"], dtype=float)
feat_cols = list(bundle["feature_columns"])
N, P = X_vals.shape
print(f"Cohort matrix: {N} x {P}")

# Build background (50 representative samples) and explainer
np.random.seed(42)
bg_idx = np.random.choice(N, 50, replace=False)
X_bg = X_vals[bg_idx]
explainer = shap.PermutationExplainer(rsf.predict, X_bg)
print("PermutationExplainer ready")

# Compute SHAP for all 546 patients in batches (timing diagnostic)
print("Computing SHAP for all 546 patients...")
t0 = time.time()
shap_obj = explainer(X_vals)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s ({elapsed/N:.2f}s per patient)")

shap_values = shap_obj.values    # (N, P)
base_value = float(shap_obj.base_values[0])
print(f"Cohort base risk = {base_value:.3f}")

# Save raw SHAP values for downstream use
np.savez_compressed(
    EXP / "shap_values_cohort.npz",
    shap_values=shap_values,
    X_vals=X_vals,
    feature_columns=np.array(feat_cols, dtype=object),
    base_value=base_value,
    bg_indices=bg_idx,
)
print(f"Saved -> {EXP / 'shap_values_cohort.npz'}")

# --------------------------------------------------------------------------
# Pretty feature names for plotting
# --------------------------------------------------------------------------
def pretty(name: str) -> str:
    if name == "pca_num_1": return "PCA 1 — Post-op alignment"
    if name == "pca_num_2": return "PCA 2 — Operative burden"
    if name == "pca_num_3": return "PCA 3 — Sagittal mismatch"
    return name.replace("_", " ").strip()

pretty_names = [pretty(c) for c in feat_cols]

# --------------------------------------------------------------------------
# FIG 14 — Beeswarm summary plot (top 15 features)
# --------------------------------------------------------------------------
top_n = 15
mean_abs = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs)[-top_n:][::-1]
top_features = [pretty_names[i] for i in top_idx]

fig, ax = plt.subplots(figsize=(10, 7))
# Use shap's beeswarm by passing pretty feature names via Explanation
exp_pretty = shap.Explanation(
    values=shap_values[:, top_idx],
    base_values=np.full(N, base_value),
    data=X_vals[:, top_idx],
    feature_names=top_features,
)
plt.sca(ax)
shap.plots.beeswarm(exp_pretty, max_display=top_n, show=False, color_bar=True)
plt.title("SHAP feature contributions across the cohort (top 15 by mean |SHAP|)",
          fontsize=12, pad=14)
plt.xlabel("SHAP value (contribution to RSF risk score)")
plt.tight_layout()
plt.savefig(FIG / "fig14_shap_summary.png")
plt.close(fig)
print("fig14 saved")


# --------------------------------------------------------------------------
# FIG 15 — Global mean |SHAP| bar (more compact alternative view)
# --------------------------------------------------------------------------
order = np.argsort(mean_abs)[-20:]   # top 20, ascending so highest at top
labels = [pretty_names[i] for i in order]
vals = mean_abs[order]

fig, ax = plt.subplots(figsize=(8.5, 6.5))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(order)))
ax.barh(range(len(order)), vals, color=colors, edgecolor="white")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels)
for i, v in enumerate(vals):
    ax.text(v + max(vals) * 0.01, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("Mean |SHAP value| across cohort")
ax.set_title("Global feature importance from cohort-wide SHAP analysis (top 20)")
ax.set_xlim(0, max(vals) * 1.13)
ax.grid(axis="x", alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIG / "fig15_shap_global_bar.png")
plt.close(fig)
print("fig15 saved")

print(f"\nAll SHAP outputs in {FIG}/")
