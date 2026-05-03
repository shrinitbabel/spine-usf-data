"""
Manuscript-ready versions of the survSHAP(t) figures using clinical labels —
the same plain-language treatment as fig14_shap_summary.png.

Reads cached experiments/survshap_global_importance.csv (and the static-SHAP
cache shap_values_cohort.npz for fig35) so no SHAP recompute is needed.

Overwrites:
  fig32_survshap_global_over_time.png
  fig33_survshap_temporal_top5.png
  fig35_static_vs_survshap_rank.png
  fig36_rank_shift_12_to_60.png
  fig37_early_vs_late_drivers.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
FIG = ROOT / "figures"

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Plain-language labels matching fig14 conventions.
PRETTY = {
    "pca_num_1": "PCA 1 — post-op alignment",
    "pca_num_2": "PCA 2 — operative burden",
    "pca_num_3": "PCA 3 — sagittal mismatch",
    "Age": "Patient age",
    "Sex": "Male sex",
    "(1 = PI>50)": "High pelvic incidence (PI > 50°)",
    "Anterior + Posterior Apporoach": "Combined anterior + posterior approach",
    "Perc screws?": "Percutaneous pedicle-screw fixation",
    "Standalone XLIF Check": "Standalone lateral interbody fusion",
    "Open Check V2": "Open posterior approach",
    "Open": "Open approach",
    "Retroperitoneal Approach (LLIF ± ALIF)": "Retroperitoneal LLIF / ALIF",
    "Osteotomies (yes/no)": "Three-column osteotomy performed",
    "ACR (y=1)": "Anterior column release",
    "prior back surgeries? (y=1)": "Prior lumbar surgery",
    "length of hospital stay (d)": "Length of hospital stay",
    "dx_spondylolisthesis": "Pre-op diagnosis: spondylolisthesis",
    "dx_spondylosis": "Pre-op diagnosis: spondylosis",
    "dx_stenosis": "Pre-op diagnosis: spinal stenosis",
    "dx_scoliosis": "Pre-op diagnosis: scoliosis",
    "dx_flat_back": "Pre-op diagnosis: flat-back deformity",
    "dx_sagittal_imbalance": "Pre-op diagnosis: sagittal imbalance",
    "dx_deformity": "Pre-op diagnosis: composite deformity",
    "dx_post_laminectomy": "Pre-op diagnosis: post-laminectomy",
    "dx_adjacent_segment": "Pre-op diagnosis: adjacent-segment disease",
    "PI-LL Mismatch Category (1 = mismatch > +/- 10": "PI–LL mismatch > 10°",
    "PI-LL Mismatch Category (1 = mismatch > +/- 9": "PI–LL mismatch > 9°",
    "Additional Procedures w/in surgery_perc screws": "Additional intra-op procedures",
    "acute thigh paresthesia (immediate post op)": "Acute post-op thigh paresthesia",
    "infection 1=yes": "Post-op infection",
}
def pretty(name: str) -> str:
    return PRETTY.get(name, name)


imp = pd.read_csv(EXP / "survshap_global_importance.csv", index_col=0)
HORIZON_COLS = ["12mo", "24mo", "36mo", "48mo", "60mo"]
HORIZON_VALS = [12, 24, 36, 48, 60]


# ============================================================ fig32 — heatmap
TOP = 12
top_feats = imp["mean"].nlargest(TOP).index.tolist()
H = imp.loc[top_feats, HORIZON_COLS].values
fig, ax = plt.subplots(figsize=(9.5, 6.5))
im = ax.imshow(H, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(HORIZON_VALS)))
ax.set_xticklabels([f"{h} mo" for h in HORIZON_VALS])
ax.set_yticks(range(TOP))
ax.set_yticklabels([pretty(f) for f in top_feats])
ax.set_xlabel("Follow-up horizon")
ax.set_title("Time-dependent feature importance  (mean |survSHAP(t)|)", pad=14)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Mean |SHAP|", rotation=270, labelpad=14)
fig.savefig(FIG / "fig32_survshap_global_over_time.png")
plt.close(fig)
print("fig32 saved")


# ============================================================ fig33 — top-5 lines
TOP5 = imp["mean"].nlargest(5).index.tolist()
fig, ax = plt.subplots(figsize=(9.5, 5.5))
palette = plt.cm.tab10(np.linspace(0, 1, len(TOP5)))
for f, col in zip(TOP5, palette):
    ax.plot(HORIZON_VALS, imp.loc[f, HORIZON_COLS].values,
            marker="o", linewidth=2.2, color=col, label=pretty(f))
ax.set_xticks(HORIZON_VALS)
ax.set_xlabel("Follow-up horizon (months)")
ax.set_ylabel("Mean |SHAP|")
ax.set_title("Temporal evolution of the top-5 risk drivers", pad=14)
ax.legend(loc="best", fontsize=9, frameon=False)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
fig.savefig(FIG / "fig33_survshap_temporal_top5.png")
plt.close(fig)
print("fig33 saved")


# ============================================================ fig36 — bump chart
ranks_t = imp[HORIZON_COLS].rank(ascending=False)
TOPN = 12
keep = ranks_t["12mo"].nsmallest(TOPN).index.union(ranks_t["60mo"].nsmallest(TOPN).index)
keep_df = ranks_t.loc[keep]
fig, ax = plt.subplots(figsize=(12, 7.5))
palette = plt.cm.tab20(np.linspace(0, 1, len(keep_df)))
for (feat, row), col in zip(keep_df.iterrows(), palette):
    ax.plot(HORIZON_VALS, row.values, "-o", color=col, lw=2)
    ax.annotate(pretty(feat), xy=(60.5, row.values[-1]),
                fontsize=9, va="center", color=col)
ax.set_xticks(HORIZON_VALS)
ax.set_xlabel("Follow-up horizon (months)")
ax.set_ylabel("Feature rank (1 = most important)")
ax.invert_yaxis()
ax.set_ylim(TOPN + 8, 0.5)
ax.set_xlim(8, 165)
ax.set_title("Feature ranking shifts across follow-up horizons (bump chart)", pad=14)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
fig.savefig(FIG / "fig36_rank_shift_12_to_60.png")
plt.close(fig)
print("fig36 saved")


# ============================================================ fig37 — early vs late
delta = (imp["60mo"] - imp["12mo"]) / imp["mean"]
delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
keep_feats = imp[imp["mean"] >= imp["mean"].quantile(0.75)].index
delta = delta.loc[delta.index.intersection(keep_feats)]
delta_sorted = delta.sort_values()
fig, ax = plt.subplots(figsize=(10.5, 8.5))
colors = ["#dc2626" if v < 0 else "#059669" for v in delta_sorted.values]
ax.barh(np.arange(len(delta_sorted)), delta_sorted.values, color=colors, alpha=0.85)
ax.set_yticks(np.arange(len(delta_sorted)))
ax.set_yticklabels([pretty(f) for f in delta_sorted.index])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Late-driver index   (60 mo − 12 mo) / mean(|SHAP|)")
ax.set_title("Early vs late drivers of ASD across the follow-up window", pad=14)
ax.text(0.02, 0.98, "← EARLY drivers (peak < 24 mo)",
        transform=ax.transAxes, fontsize=10, va="top",
        color="#dc2626", weight="bold")
ax.text(0.98, 0.98, "LATE drivers (peak > 36 mo) →",
        transform=ax.transAxes, fontsize=10, va="top", ha="right",
        color="#059669", weight="bold")
ax.grid(True, axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
fig.savefig(FIG / "fig37_early_vs_late_drivers.png")
plt.close(fig)
print("fig37 saved")


# ============================================================ fig35 — rank comparison
# Reuse the cached cohort static SHAP cache (from shap_summary_publication.py)
# rather than recomputing — same numbers, no wait.
import pickle
sv_path = EXP / "shap_values_cohort.npz"
if sv_path.exists():
    d = np.load(sv_path, allow_pickle=True)
    sv_static = np.asarray(d["shap_values"])
    fc_static = list(d["feature_columns"])
    static_imp = pd.Series(np.abs(sv_static).mean(axis=0), index=fc_static, name="static")
else:
    print("WARN: shap_values_cohort.npz missing — falling back to a fresh static SHAP run")
    import shap
    with open(ROOT / "models" / "rsf_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    rsf = bundle["model"]; X_b = bundle["X_vals"]; fc_static = bundle["feature_columns"]
    rng = np.random.default_rng(42)
    bg = rng.choice(len(X_b), 50, replace=False)
    sm = rng.choice(len(X_b), 120, replace=False)
    exp_static = shap.PermutationExplainer(rsf.predict, X_b[bg])
    sv_static = exp_static(X_b[sm], max_evals=400).values
    static_imp = pd.Series(np.abs(sv_static).mean(axis=0), index=fc_static, name="static")

static_rank = static_imp.rank(ascending=False).rename("static_rank")
surv_rank = imp["mean"].rank(ascending=False).rename("survSHAP_rank")
rank_df = pd.concat([static_rank, surv_rank], axis=1).dropna()
TOPN = 15
top_static = rank_df.nsmallest(TOPN, "static_rank").index.tolist()
top_surv = rank_df.nsmallest(TOPN, "survSHAP_rank").index.tolist()
union = list(dict.fromkeys(top_static + top_surv))[:20]

fig, ax = plt.subplots(figsize=(11, 8))
y = np.arange(len(union))
sr = rank_df.loc[union, "static_rank"].values
vr = rank_df.loc[union, "survSHAP_rank"].values
for i, _ in enumerate(union):
    color = "#059669" if vr[i] < sr[i] else ("#dc2626" if vr[i] > sr[i] else "#94a3b8")
    ax.plot([sr[i], vr[i]], [i, i], color=color, lw=2, alpha=0.7)
ax.scatter(sr, y, s=85, color="#94a3b8", label="Static SHAP rank", zorder=3)
ax.scatter(vr, y, s=85, color="#059669", label="survSHAP(t) rank (time-avg)", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([pretty(f) for f in union])
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_xlabel("Rank (1 = most important)")
ax.set_title("Static SHAP vs time-dependent survSHAP(t) feature ranking", pad=14)
ax.legend(loc="lower right", frameon=False)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
fig.savefig(FIG / "fig35_static_vs_survshap_rank.png")
plt.close(fig)
print("fig35 saved")
