"""
Generate publication figures (300 DPI) for the manuscript.

All figures saved to figures/. Reuses pre-computed CV outputs in
experiments/ to keep runtime under a minute.

Figures:
  --- Dataset (5) ---
  fig01_cohort_overview.png       Overall KM survival curve
  fig02_demographics.png           Age / BMI / Sex / Prior surgery panel
  fig03_event_timing.png           Distribution of event vs censoring times
  fig04_surgical_profile.png       Levels fused / approach / diagnosis breakdown
  fig05_alignment_by_outcome.png   Sagittal alignment (PI-LL, SVA, PT, LL) vs ASD status

  --- Model performance (7) ---
  fig06_cindex_per_fold.png        Per-fold C-index, all 3 deployed models
  fig07_time_dep_auc.png           RSF time-dependent AUC (mean +/- std)
  fig08_calibration_RSF.png        4-panel calibration plot @ 12/24/36/60 mo with ECE
  fig09_feature_importance.png     Top 20 RSF feature importances
  fig10_pca_loadings.png           PCA component loadings heatmap
  fig11_preset_comparison.png      8 clinical presets, 5y survival per model
  fig12_model_concordance.png      Spearman concordance between model OOF risks
"""

from __future__ import annotations
import json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines import KaplanMeierFitter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Publication style
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# Consistent palette across model figures
PALETTE = {"RSF": "#2563eb", "DeepSurv": "#dc2626", "Ensemble": "#7c3aed"}

# ==========================================================================
# Load data
# ==========================================================================
df = pd.read_csv(ROOT / "cleaned_data.csv")
df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)
df["REVERIFIED ASD"] = df["REVERIFIED ASD"].fillna(0).astype(int)
df["time_surv"] = np.where(
    df["REVERIFIED ASD"] == 1,
    df["Time Until ASD Diagnosis (months)"],
    df["Time Without_ASD (months)"],
)
df = df.dropna(subset=["time_surv"]).reset_index(drop=True)
df["time_surv"] = df["time_surv"].astype(float)

times = df["time_surv"].values
events = df["REVERIFIED ASD"].astype(bool).values
N, n_events = len(df), int(events.sum())
print(f"Cohort: {N} patients, {n_events} events ({n_events/N*100:.1f}%)")

with open(ROOT / "models" / "rsf_bundle.pkl", "rb") as f:
    rsf_bundle = pickle.load(f)

with open(EXP / "extended_results.json") as f:
    ext = json.load(f)
with open(EXP / "calibration_results.json") as f:
    cal = json.load(f)
fold_metrics = pd.read_csv(EXP / "fold_metrics.csv")
preset_preds = pd.read_csv(EXP / "preset_preds.csv")
calib_RSF = pd.read_csv(EXP / "calibration_bins_RSF.csv")


# ==========================================================================
# FIG 01 — Overall KM survival curve
# ==========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
km = KaplanMeierFitter()
km.fit(times, event_observed=events, label="Cohort (N=546)")
km.plot_survival_function(ax=ax, ci_show=True, color="#1e40af", linewidth=2)
ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel("Months since index surgery")
ax.set_ylabel("ASD-free survival probability")
ax.set_title("Cohort Kaplan–Meier curve for adjacent segment disease")
ax.set_ylim(0, 1.02)
ax.set_xlim(0, max(times) + 5)
# Annotate event count
ax.text(0.02, 0.07,
        f"Events: {n_events} / {N}  (event rate {n_events/N*100:.1f}%)\n"
        f"Median follow-up: {np.median(times):.0f} months",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1"))
plt.savefig(FIG / "fig01_cohort_overview.png")
plt.close(fig)
print("fig01 saved")


# ==========================================================================
# FIG 02 — Demographics (Age / BMI / Sex / Prior surgery)
# ==========================================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

age = pd.to_numeric(df.get("Age"), errors="coerce").dropna()
axes[0, 0].hist(age, bins=25, color="#3b82f6", edgecolor="white", alpha=0.85)
axes[0, 0].axvline(age.mean(), color="#dc2626", linestyle="--", linewidth=1.5,
                    label=f"Mean = {age.mean():.1f}")
axes[0, 0].set_xlabel("Age (years)")
axes[0, 0].set_ylabel("Patients")
axes[0, 0].set_title("Age distribution")
axes[0, 0].legend(loc="upper right")

bmi = pd.to_numeric(df.get("BMI"), errors="coerce").dropna()
axes[0, 1].hist(bmi, bins=25, color="#10b981", edgecolor="white", alpha=0.85)
axes[0, 1].axvline(bmi.mean(), color="#dc2626", linestyle="--", linewidth=1.5,
                    label=f"Mean = {bmi.mean():.1f}")
axes[0, 1].set_xlabel("BMI (kg/m²)")
axes[0, 1].set_ylabel("Patients")
axes[0, 1].set_title("BMI distribution")
axes[0, 1].legend(loc="upper right")

sex = pd.to_numeric(df.get("Sex"), errors="coerce").dropna().astype(int)
sex_counts = sex.value_counts().sort_index()
labels = ["Female" if k == 0 else "Male" for k in sex_counts.index]
axes[1, 0].bar(labels, sex_counts.values,
               color=["#fb923c", "#3b82f6"], edgecolor="white")
for i, v in enumerate(sex_counts.values):
    axes[1, 0].text(i, v + 5, f"{v}\n({v/sex_counts.sum()*100:.0f}%)",
                     ha="center", fontsize=9)
axes[1, 0].set_ylabel("Patients")
axes[1, 0].set_title("Sex distribution")
axes[1, 0].set_ylim(0, max(sex_counts.values) * 1.18)

prior = pd.to_numeric(df.get("prior back surgeries? (y=1)"),
                       errors="coerce").fillna(0).astype(int)
prior_counts = prior.value_counts().sort_index()
labels2 = ["Primary fusion", "Prior surgery"]
axes[1, 1].bar(labels2, prior_counts.values,
               color=["#94a3b8", "#dc2626"], edgecolor="white")
for i, v in enumerate(prior_counts.values):
    axes[1, 1].text(i, v + 5, f"{v}\n({v/prior_counts.sum()*100:.0f}%)",
                     ha="center", fontsize=9)
axes[1, 1].set_ylabel("Patients")
axes[1, 1].set_title("Prior back surgery history")
axes[1, 1].set_ylim(0, max(prior_counts.values) * 1.18)

fig.suptitle("Patient demographics", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(FIG / "fig02_demographics.png")
plt.close(fig)
print("fig02 saved")


# ==========================================================================
# FIG 03 — Event timing distribution (event vs censoring)
# ==========================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
event_t = times[events]
cens_t = times[~events]
bins = np.linspace(0, max(times), 30)
ax.hist(cens_t, bins=bins, color="#94a3b8", alpha=0.65, edgecolor="white",
        label=f"Censored (n = {len(cens_t)})")
ax.hist(event_t, bins=bins, color="#dc2626", alpha=0.85, edgecolor="white",
        label=f"ASD event (n = {len(event_t)})")
ax.axvline(np.median(event_t), color="#dc2626", linestyle="--", linewidth=1.2,
           alpha=0.8, label=f"Median time-to-ASD = {np.median(event_t):.0f} mo")
ax.set_xlabel("Months since index surgery")
ax.set_ylabel("Patients")
ax.set_title("Distribution of event and censoring times")
ax.legend(loc="upper right")
plt.savefig(FIG / "fig03_event_timing.png")
plt.close(fig)
print("fig03 saved")


# ==========================================================================
# FIG 04 — Surgical profile
# ==========================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# 4a. Levels fused
level_cols = ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
levels_per_pt = sum(
    pd.to_numeric(df.get(c), errors="coerce").fillna(0).astype(int) for c in level_cols
)
lvl_counts = levels_per_pt.value_counts().sort_index()
axes[0].bar(lvl_counts.index, lvl_counts.values,
            color="#2563eb", edgecolor="white", alpha=0.85)
for i, v in zip(lvl_counts.index, lvl_counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontsize=8)
axes[0].set_xlabel("Levels fused per patient")
axes[0].set_ylabel("Patients")
axes[0].set_title("Construct length")
axes[0].set_xticks(sorted(lvl_counts.index))

# 4b. Surgical approach mix (count of patients with each technique flag)
approach_cols = {
    "Open": "Open",
    "Perc screws?": "Percutaneous",
    "Standalone XLIF Check": "Standalone XLIF",
    "Retroperitoneal Approach (LLIF ± ALIF)": "Retroperitoneal",
    "Anterior + Posterior Apporoach": "A+P approach",
    "Osteotomies (yes/no)": "Osteotomies",
}
counts = {label: int(pd.to_numeric(df.get(col), errors="coerce").fillna(0).astype(int).sum())
          for col, label in approach_cols.items()}
sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
labels = [k for k, _ in sorted_items]
vals = [v for _, v in sorted_items]
axes[1].barh(labels, vals, color="#7c3aed", edgecolor="white", alpha=0.85)
for i, v in enumerate(vals):
    axes[1].text(v + 3, i, str(v), va="center", fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlabel("Patients (any case may have multiple)")
axes[1].set_title("Surgical technique mix")
axes[1].set_xlim(0, max(vals) * 1.15)

# 4c. Pre-op diagnosis flags
dx_cols = {
    "dx_stenosis": "Stenosis",
    "dx_spondylolisthesis": "Spondylolisthesis",
    "dx_spondylosis": "Spondylosis",
    "dx_scoliosis": "Scoliosis",
    "dx_flat_back": "Flat-back",
    "dx_sagittal_imbalance": "Sagittal imbalance",
    "dx_deformity": "Deformity (composite)",
}
dx_counts = {label: int(pd.to_numeric(df.get(col), errors="coerce").fillna(0).astype(int).sum())
             for col, label in dx_cols.items() if col in df.columns}
sorted_dx = sorted(dx_counts.items(), key=lambda x: x[1], reverse=True)
labels = [k for k, _ in sorted_dx]
vals = [v for _, v in sorted_dx]
axes[2].barh(labels, vals, color="#dc2626", edgecolor="white", alpha=0.85)
for i, v in enumerate(vals):
    axes[2].text(v + 3, i, str(v), va="center", fontsize=8)
axes[2].invert_yaxis()
axes[2].set_xlabel("Patients (any case may have multiple)")
axes[2].set_title("Pre-operative diagnoses")
axes[2].set_xlim(0, max(vals) * 1.15)

fig.suptitle("Surgical profile of the cohort", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "fig04_surgical_profile.png")
plt.close(fig)
print("fig04 saved")


# ==========================================================================
# FIG 05 — Sagittal alignment by outcome (boxplots)
# ==========================================================================
fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))

align_features = [
    ("ABS PI-LL angle mismatch", "PI–LL mismatch (°)"),
    ("post SVA", "Post-op SVA (mm)"),
    ("post PT", "Post-op PT (°)"),
    ("post LL", "Post-op LL (°)"),
]
for ax, (col, label) in zip(axes, align_features):
    vals = pd.to_numeric(df[col], errors="coerce")
    asd_yes = vals[events].dropna().values
    asd_no = vals[~events].dropna().values
    bp = ax.boxplot([asd_no, asd_yes], labels=["No ASD", "ASD"],
                    patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=4, alpha=0.5))
    bp["boxes"][0].set_facecolor("#94a3b8")
    bp["boxes"][1].set_facecolor("#dc2626")
    for box in bp["boxes"]:
        box.set_alpha(0.75); box.set_edgecolor("white")
    ax.set_ylabel(label)
    ax.set_title(label.split(" (")[0])

fig.suptitle("Sagittal alignment metrics by ASD outcome", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "fig05_alignment_by_outcome.png")
plt.close(fig)
print("fig05 saved")


# ==========================================================================
# FIG 06 — Per-fold C-index across the three deployed models
# ==========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))

models_to_plot = ["RSF", "DeepSurv", "Ensemble"]
fold_data = {m: fold_metrics[f"{m}_cindex"].dropna().values for m in models_to_plot}

positions = np.arange(len(models_to_plot))
bp = ax.boxplot([fold_data[m] for m in models_to_plot],
                positions=positions, widths=0.5, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=5, alpha=0.6))
for box, m in zip(bp["boxes"], models_to_plot):
    box.set_facecolor(PALETTE[m]); box.set_alpha(0.65); box.set_edgecolor("white")

# Strip plot of individual folds
for i, m in enumerate(models_to_plot):
    jitter = np.random.RandomState(42).normal(0, 0.05, size=len(fold_data[m]))
    ax.scatter(positions[i] + jitter, fold_data[m], color=PALETTE[m],
               s=30, edgecolor="white", linewidth=0.6, alpha=0.85, zorder=3)

# Means with text labels
for i, m in enumerate(models_to_plot):
    mean_v = np.mean(fold_data[m])
    ax.scatter(positions[i], mean_v, color="black", marker="D", s=40, zorder=4,
               label="Mean" if i == 0 else None)
    ax.text(positions[i] + 0.30, mean_v, f"{mean_v:.3f}",
            va="center", fontsize=9, fontweight="bold")

ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.7,
           label="Chance (C = 0.5)")
ax.set_xticks(positions)
ax.set_xticklabels(models_to_plot)
ax.set_ylabel("C-index (per fold)")
ax.set_title("Per-fold discrimination (10-fold honest cross-validation)")
ax.set_ylim(0.4, 0.95)
ax.legend(loc="upper right")
plt.savefig(FIG / "fig06_cindex_per_fold.png")
plt.close(fig)
print("fig06 saved")


# ==========================================================================
# FIG 07 — Time-dependent AUC for RSF
# ==========================================================================
auc_rows = ext.get("rsf_time_auc", [])
horizons = [r["horizon_months"] for r in auc_rows]
auc_means = [r["auc_mean"] for r in auc_rows]
auc_stds = [r["auc_std"] for r in auc_rows]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(horizons, auc_means, color=PALETTE["RSF"], marker="o", linewidth=2,
        markersize=7, label="Mean time-dependent AUC")
ax.fill_between(horizons,
                np.maximum(0, np.array(auc_means) - np.array(auc_stds)),
                np.minimum(1, np.array(auc_means) + np.array(auc_stds)),
                color=PALETTE["RSF"], alpha=0.18, label="±1 SD across folds")
ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.7,
           label="Chance (AUC = 0.5)")
for h, m in zip(horizons, auc_means):
    ax.annotate(f"{m:.2f}", xy=(h, m), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Horizon (months post-surgery)")
ax.set_ylabel("Time-dependent AUC")
ax.set_title("RSF time-dependent AUC by prediction horizon")
ax.set_ylim(0.3, 0.85)
ax.set_xticks(horizons)
ax.legend(loc="upper right")
plt.savefig(FIG / "fig07_time_dep_auc.png")
plt.close(fig)
print("fig07 saved")


# ==========================================================================
# FIG 08 — Calibration plots for RSF at 12 / 24 / 36 / 60 mo
# ==========================================================================
ece_RSF = cal["ece"]["RSF"]
horizons_to_plot = [12, 24, 36, 60]

fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
for ax, h in zip(axes.flatten(), horizons_to_plot):
    sub = calib_RSF[calib_RSF["horizon"] == h].sort_values("pred_mean")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, alpha=0.7,
            label="Perfect calibration")
    ax.plot(sub["pred_mean"], sub["obs"], marker="o", color=PALETTE["RSF"],
            linewidth=2, markersize=7, label="RSF (KM-binned)")
    ax.scatter(sub["pred_mean"], sub["obs"], s=sub["n"] * 4,
               color=PALETTE["RSF"], edgecolor="white", linewidth=0.8,
               alpha=0.85, zorder=3)
    ax.set_xlabel("Mean predicted ASD-free probability")
    ax.set_ylabel("Observed (Kaplan–Meier)")
    ax.set_title(f"{h}-month horizon  ·  ECE = {ece_RSF.get(str(h), 0):.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)

fig.suptitle("RSF calibration by prediction horizon (point size ∝ patients per bin)",
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(FIG / "fig08_calibration_RSF.png")
plt.close(fig)
print("fig08 saved")


# ==========================================================================
# FIG 09 — RSF feature importances (top 20)
# ==========================================================================
rsf_model = rsf_bundle["model"]
feat_cols = rsf_bundle["feature_columns"]

# sksurv's RandomSurvivalForest does not implement feature_importances_.
# Use split-count importance: for each feature, count how often it is used
# as a split decision across all 200 trees. Normalize to sum to 1.
n_feat = len(feat_cols)
split_counts = np.zeros(n_feat)
for tree in rsf_model.estimators_:
    feat_used = tree.tree_.feature
    feat_used = feat_used[feat_used >= 0]   # -2 marks leaves
    np.add.at(split_counts, feat_used, 1)
if split_counts.sum() > 0:
    split_counts = split_counts / split_counts.sum()
imp = pd.Series(split_counts, index=feat_cols).sort_values(ascending=False)

# Pretty-print PCA component names
def pretty(name: str) -> str:
    if name == "pca_num_1": return "PCA 1 — Post-op alignment"
    if name == "pca_num_2": return "PCA 2 — Operative burden"
    if name == "pca_num_3": return "PCA 3 — Sagittal mismatch"
    return name

top_n = 20
top_imp = imp.head(top_n)
labels = [pretty(c)[:55] for c in top_imp.index]

fig, ax = plt.subplots(figsize=(8.5, 6))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, top_n))[::-1]
ax.barh(range(top_n), top_imp.values[::-1], color=colors, edgecolor="white")
ax.set_yticks(range(top_n))
ax.set_yticklabels(labels[::-1])
for i, v in enumerate(top_imp.values[::-1]):
    ax.text(v + max(top_imp) * 0.01, i, f"{v:.4f}", va="center", fontsize=8)
ax.set_xlabel("RSF impurity-based feature importance")
ax.set_title(f"Top {top_n} predictors in the trained RSF model")
ax.set_xlim(0, max(top_imp) * 1.15)
plt.savefig(FIG / "fig09_feature_importance.png")
plt.close(fig)
print("fig09 saved")


# ==========================================================================
# FIG 10 — PCA component loadings heatmap
# ==========================================================================
numeric_present = rsf_bundle["numeric_present"]
pca = rsf_bundle["pca"]
loadings = pd.DataFrame(pca.components_,
                         index=[f"PCA {i+1}" for i in range(pca.components_.shape[0])],
                         columns=numeric_present)

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(loadings.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(len(numeric_present)))
ax.set_xticklabels(numeric_present, rotation=35, ha="right")
ax.set_yticks(range(loadings.shape[0]))
ax.set_yticklabels(loadings.index)
# Annotate values
for i in range(loadings.shape[0]):
    for j in range(loadings.shape[1]):
        v = loadings.iat[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(v) > 0.35 else "black")
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Loading")
ax.set_title("PCA component loadings on continuous alignment / surgical features")
ax.grid(False)
plt.savefig(FIG / "fig10_pca_loadings.png")
plt.close(fig)
print("fig10 saved")


# ==========================================================================
# FIG 11 — 8 clinical presets, 5y survival per model (grouped bar)
# ==========================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
preset_order = (
    preset_preds[preset_preds["model"] == "RSF"]
    .sort_values("asd_free_5y")["preset"].tolist()
)
x = np.arange(len(preset_order))
width = 0.27
for i, m in enumerate(["RSF", "DeepSurv", "Coxnet"]):
    sub = preset_preds[preset_preds["model"] == m].set_index("preset").reindex(preset_order)
    ax.bar(x + (i - 1) * width, sub["asd_free_5y"] * 100, width,
           label=m, color=PALETTE.get(m, "#10b981"), edgecolor="white", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(preset_order, rotation=30, ha="right")
ax.set_ylabel("5-year ASD-free probability (%)")
ax.set_title("Predicted 5-year survival across 8 clinical presets, by model")
ax.axhline(78, color="grey", linestyle=":", linewidth=1, alpha=0.7,
           label="Cohort baseline (≈78% ASD-free)")
ax.legend(loc="upper right", ncol=4)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(FIG / "fig11_preset_comparison.png")
plt.close(fig)
print("fig11 saved")


# ==========================================================================
# FIG 12 — Spearman concordance between model risk scores
# ==========================================================================
corr = ext.get("spearman_corr", {})
if corr:
    cdf = pd.DataFrame(corr)
    # Ensure matching ordering
    cdf = cdf.loc[cdf.columns]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cdf.values, cmap="YlGnBu", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(cdf))); ax.set_xticklabels(cdf.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(cdf))); ax.set_yticklabels(cdf.index)
    for i in range(cdf.shape[0]):
        for j in range(cdf.shape[1]):
            v = cdf.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.85 else "black")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Spearman ρ")
    ax.set_title("Inter-model agreement on out-of-fold patient risks")
    ax.grid(False)
    plt.savefig(FIG / "fig12_model_concordance.png")
    plt.close(fig)
    print("fig12 saved")

# ==========================================================================
# FIG 13 — All base models: OOF C-index (the "model selection" figure)
# ==========================================================================
oof = ext.get("oof_cindex", {})
ens = ext.get("ensembles", {})

# Individual models with full names
indiv_label = {
    "RSF": "Random Survival Forest",
    "DeepSurv": "DeepSurv (CoxPH neural net)",
    "GBSA": "Gradient Boosting Survival",
    "CWGB": "Componentwise Gradient Boost",
    "Coxnet": "Cox elastic-net",
}
indiv_palette = {
    "RSF": "#2563eb", "DeepSurv": "#dc2626", "GBSA": "#f59e0b",
    "CWGB": "#10b981", "Coxnet": "#64748b",
}

# Ensemble entry: rank-averaged RSF + DeepSurv (the one we deployed)
deployed_ens_key = "RSF+DeepSurv"
deployed_ens_C = ens.get(deployed_ens_key, np.nan)

# Build sorted bar list
bars = [(indiv_label[k], oof[k], indiv_palette[k], False) for k in indiv_label if k in oof]
bars.append(("Ensemble (RSF + DeepSurv)", deployed_ens_C, "#7c3aed", True))
bars.sort(key=lambda x: x[1])  # ascending so highest C is at the top of horizontal bars

fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                          gridspec_kw={"width_ratios": [1.4, 1]})

# Left panel — all models, OOF C-index
ax = axes[0]
ypos = np.arange(len(bars))
for i, (label, c, color, is_ens) in enumerate(bars):
    ax.barh(i, c, color=color, edgecolor="white", alpha=0.9,
            hatch="//" if is_ens else None)
    ax.text(c + 0.005, i, f"{c:.3f}", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(ypos)
ax.set_yticklabels([b[0] for b in bars])
ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.7,
           label="Chance (C = 0.5)")
ax.axvline(max(b[1] for b in bars), color="#2563eb", linestyle="--",
           linewidth=1, alpha=0.5, label="Best single model")
ax.set_xlim(0.5, max(b[1] for b in bars) + 0.05)
ax.set_xlabel("Out-of-fold concordance index")
ax.set_title("(A) All evaluated survival models — pooled OOF C-index")
ax.legend(loc="lower right", fontsize=8)

legend_patches = [
    Patch(facecolor="#94a3b8", edgecolor="white", label="Individual model"),
    Patch(facecolor="#7c3aed", edgecolor="white", hatch="//", label="Ensemble (rank-average)"),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color="grey", linestyle=":", label="Chance (C = 0.5)"),
], loc="lower right", fontsize=8)

# Right panel — ensembling failed to improve over RSF
ax = axes[1]
ens_keys = [
    "RSF",
    "RSF+DeepSurv",
    "RSF+GBSA",
    "RSF+CWGB",
    "RSF+Coxnet",
    "RSF+GBSA+DeepSurv",
    "RSF+GBSA+Coxnet+CWGB+DeepSurv",
]
labels_short = [
    "RSF alone",
    "+ DeepSurv",
    "+ GBSA",
    "+ CWGB",
    "+ Coxnet",
    "+ DeepSurv + GBSA",
    "All five",
]
ens_lookup = {**{"RSF": oof.get("RSF")}, **ens}
vals = [ens_lookup.get(k, np.nan) for k in ens_keys]

ypos = np.arange(len(ens_keys))
colors = ["#2563eb"] + ["#7c3aed"] * (len(ens_keys) - 1)
for i, (lab, v, color) in enumerate(zip(labels_short, vals, colors)):
    ax.barh(i, v, color=color, edgecolor="white", alpha=0.85,
            hatch="//" if i > 0 else None)
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(ypos)
ax.set_yticklabels(labels_short)
ax.invert_yaxis()
ax.axvline(oof["RSF"], color="#2563eb", linestyle="--", linewidth=1.2,
           alpha=0.7, label="RSF alone (reference)")
ax.set_xlim(0.55, max(v for v in vals if not np.isnan(v)) + 0.02)
ax.set_xlabel("OOF concordance index")
ax.set_title("(B) Ensembling — no improvement over RSF alone")
ax.legend(loc="lower right", fontsize=8)

fig.suptitle("Model selection — discrimination across all candidate survival models",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "fig13_all_models_comparison.png")
plt.close(fig)
print("fig13 saved")


print(f"\nAll figures saved to {FIG}/")
print("\nFile listing:")
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
