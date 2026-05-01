"""
Post-hoc analysis of survSHAP(t) results - works from
experiments/survshap_global_importance.csv only (no re-run needed).

Generates:
  fig35_static_vs_survshap_rank.png  - rank-comparison instead of broken raw-magnitude
  fig36_rank_shift_12_to_60.png      - bump chart of feature rank changes
  fig37_early_vs_late_drivers.png    - bar chart of early-driver / late-driver index
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle, shap

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"

imp = pd.read_csv(ROOT / "experiments" / "survshap_global_importance.csv", index_col=0)
HORIZON_COLS = ["12mo","24mo","36mo","48mo","60mo"]

# --------------------------------------------------------------------------
# Compute static SHAP for fig35 rank comparison
# --------------------------------------------------------------------------
with open(ROOT / "models" / "rsf_bundle.pkl","rb") as f:
    bundle = pickle.load(f)
rsf = bundle["model"]; X = bundle["X_vals"]; fc = bundle["feature_columns"]
rng = np.random.default_rng(42)
bg = rng.choice(len(X), 50, replace=False)
sm = rng.choice(len(X), 120, replace=False)
exp_static = shap.PermutationExplainer(rsf.predict, X[bg])
sv = exp_static(X[sm], max_evals=400).values
static_imp = pd.Series(np.abs(sv).mean(axis=0), index=fc, name="static")

# --------------------------------------------------------------------------
# fig35: rank-based comparison (corrected)
# --------------------------------------------------------------------------
static_rank = static_imp.rank(ascending=False).rename("static_rank")
surv_rank = imp["mean"].rank(ascending=False).rename("survSHAP_rank")
rank_df = pd.concat([static_rank, surv_rank], axis=1).dropna()
TOPN = 15
top_static = rank_df.nsmallest(TOPN, "static_rank").index.tolist()
top_surv = rank_df.nsmallest(TOPN, "survSHAP_rank").index.tolist()
union = list(dict.fromkeys(top_static + top_surv))[:20]

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(union))
sr = rank_df.loc[union, "static_rank"].values
vr = rank_df.loc[union, "survSHAP_rank"].values
for i, f in enumerate(union):
    color = "#3a7" if vr[i] < sr[i] else ("#c63" if vr[i] > sr[i] else "#888")
    ax.plot([sr[i], vr[i]], [i, i], color=color, lw=2, alpha=0.6)
ax.scatter(sr, y, s=80, color="#888", label="static SHAP rank", zorder=3)
ax.scatter(vr, y, s=80, color="#3a7", label="survSHAP(t) rank (time-avg)", zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f[:42] for f in union], fontsize=9)
ax.invert_yaxis(); ax.invert_xaxis()
ax.set_xlabel("Rank (1 = most important)")
ax.set_title("Static SHAP vs time-dependent SHAP: feature ranking")
ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "fig35_static_vs_survshap_rank.png", dpi=180)
plt.close(fig)
print("wrote fig35_static_vs_survshap_rank.png")

# --------------------------------------------------------------------------
# fig36: rank shift 12mo -> 60mo (bump chart)
# --------------------------------------------------------------------------
ranks_t = imp[HORIZON_COLS].rank(ascending=False)
TOPN = 12
keep = ranks_t["12mo"].nsmallest(TOPN).index.union(ranks_t["60mo"].nsmallest(TOPN).index)
keep_df = ranks_t.loc[keep]

fig, ax = plt.subplots(figsize=(10, 7))
xs = np.arange(len(HORIZON_COLS))
horizon_vals = [12, 24, 36, 48, 60]
palette = plt.cm.tab20(np.linspace(0, 1, len(keep_df)))
for (feat, row), col in zip(keep_df.iterrows(), palette):
    ax.plot(horizon_vals, row.values, "-o", color=col, lw=2, label=feat[:38])
    ax.annotate(feat[:30], xy=(60.5, row.values[-1]), fontsize=8, va="center", color=col)
ax.set_xticks(horizon_vals); ax.set_xlabel("Follow-up horizon (months)")
ax.set_ylabel("Feature rank (1 = most important)")
ax.invert_yaxis(); ax.set_ylim(TOPN+8, 0.5)
ax.set_xlim(8, 95)
ax.set_title("Feature ranking shifts across follow-up horizons (bump chart)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "fig36_rank_shift_12_to_60.png", dpi=180)
plt.close(fig)
print("wrote fig36_rank_shift_12_to_60.png")

# --------------------------------------------------------------------------
# fig37: early-vs-late driver index = (60mo - 12mo) / mean
# Positive = late driver, negative = early driver
# --------------------------------------------------------------------------
delta = (imp["60mo"] - imp["12mo"]) / imp["mean"]
delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
# Restrict to features with non-trivial importance
keep_feats = imp[imp["mean"] >= imp["mean"].quantile(0.75)].index
delta = delta.loc[delta.index.intersection(keep_feats)]
delta_sorted = delta.sort_values()

fig, ax = plt.subplots(figsize=(9, 8))
colors = ["#c63" if v < 0 else "#3a7" for v in delta_sorted.values]
ax.barh(np.arange(len(delta_sorted)), delta_sorted.values, color=colors)
ax.set_yticks(np.arange(len(delta_sorted)))
ax.set_yticklabels([f[:42] for f in delta_sorted.index], fontsize=9)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Late-driver index  (60mo - 12mo) / mean(|SHAP|)")
ax.set_title("Early- vs late-driver classification of top features")
ax.text(0.02, 0.98, "← EARLY drivers (peak <24mo)", transform=ax.transAxes,
        fontsize=10, va="top", color="#c63", weight="bold")
ax.text(0.98, 0.98, "LATE drivers (peak >36mo) →", transform=ax.transAxes,
        fontsize=10, va="top", ha="right", color="#3a7", weight="bold")
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "fig37_early_vs_late_drivers.png", dpi=180)
plt.close(fig)
print("wrote fig37_early_vs_late_drivers.png")

# Summary table
summary = pd.DataFrame({
    "12mo_imp": imp["12mo"],
    "60mo_imp": imp["60mo"],
    "delta_pct": (imp["60mo"] - imp["12mo"]) / imp["12mo"] * 100,
    "12mo_rank": ranks_t["12mo"].astype(int),
    "60mo_rank": ranks_t["60mo"].astype(int),
    "rank_shift_12_to_60": ranks_t["12mo"].astype(int) - ranks_t["60mo"].astype(int),
}).loc[imp["mean"].nlargest(15).index]
print("\n--- top-15 features: early vs late driver summary ---")
print(summary.round(3).to_string())
summary.to_csv(ROOT / "experiments" / "survshap_early_vs_late_summary.csv")
