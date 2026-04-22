"""
SHAP dependence plots for the trained RSF, generated as individual
publication figures (300 DPI). One PNG per plot — NO subplots — so
they can be inserted independently into the manuscript.

Each plot shows:
  x-axis: feature value
  y-axis: SHAP contribution to RSF risk score
  color : a second (interacting) feature

Three-way plots are produced as side-by-side faceted figures (e.g.,
PCA1 vs SHAP colored by Age, faceted by PI>50 high vs low).

All inputs come from `experiments/shap_values_cohort.npz` (precomputed)
plus original `cleaned_data.csv` for any continuous feature that was
absorbed into the PCA pipeline (post-SVA, post-PT, etc.).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

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
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
d = np.load(EXP / "shap_values_cohort.npz", allow_pickle=True)
SV = d["shap_values"]          # (N, P)
X = d["X_vals"]                # (N, P)
feats = list(d["feature_columns"])
base = float(d["base_value"])
N, P = SV.shape

# Feature index lookup
fi = {name: i for i, name in enumerate(feats)}

# Original cleaned_data — used to recover continuous features that PCA absorbed
df_raw = pd.read_csv(ROOT / "cleaned_data.csv")
df_raw.columns = df_raw.columns.str.strip().str.replace("\n", " ", regex=False)
df_raw["REVERIFIED ASD"] = df_raw["REVERIFIED ASD"].fillna(0).astype(int)
df_raw["time_surv"] = np.where(
    df_raw["REVERIFIED ASD"] == 1,
    df_raw["Time Until ASD Diagnosis (months)"],
    df_raw["Time Without_ASD (months)"],
)
df_raw = df_raw.dropna(subset=["time_surv"]).reset_index(drop=True)

def raw_col(name: str) -> np.ndarray:
    return pd.to_numeric(df_raw[name], errors="coerce").values

# Levels fused proxy from binary level columns
lvl_cols = ["T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"]
levels_per_pt = sum(
    pd.to_numeric(df_raw[c], errors="coerce").fillna(0).astype(int) for c in lvl_cols
).values

# Pretty names for axis labels
PRETTY = {
    "pca_num_1": "PCA 1 — Post-op sagittal alignment composite",
    "pca_num_2": "PCA 2 — Operative burden composite",
    "pca_num_3": "PCA 3 — Sagittal mismatch composite",
    "Age": "Patient age (years)",
    "(1 = PI>50)": "Pelvic incidence > 50°",
    "Anterior + Posterior Apporoach": "Anterior + posterior approach",
    "prior back surgeries? (y=1)": "Prior back surgery",
    "Perc screws?": "Percutaneous screws",
    "dx_spondylolisthesis": "Diagnosis: spondylolisthesis",
    "dx_flat_back": "Diagnosis: flat-back deformity",
    "dx_scoliosis": "Diagnosis: scoliosis",
    "dx_stenosis": "Diagnosis: spinal stenosis",
}
def label(name: str) -> str:
    return PRETTY.get(name, name)


# ---------------------------------------------------------------------
# Helper: continuous-color dependence plot
# ---------------------------------------------------------------------
def dep_plot(
    x_name, color_data, color_label, fname, *,
    color_kind="continuous", title_extra="",
    cmap="viridis", color_vmin=None, color_vmax=None,
    figsize=(8, 5), x_jitter=0.0,
):
    """Save one dependence plot to figures/<fname>."""
    if x_name not in fi:
        print(f"  [skip] feature '{x_name}' not in model"); return
    xi = fi[x_name]
    x = X[:, xi].astype(float).copy()
    y = SV[:, xi]
    if x_jitter > 0:
        x = x + np.random.RandomState(42).normal(0, x_jitter, size=len(x))

    fig, ax = plt.subplots(figsize=figsize)

    if color_kind == "continuous":
        valid = ~np.isnan(color_data) & ~np.isnan(x) & ~np.isnan(y)
        norm = Normalize(
            vmin=color_vmin if color_vmin is not None else np.nanpercentile(color_data, 5),
            vmax=color_vmax if color_vmax is not None else np.nanpercentile(color_data, 95),
        )
        sc = ax.scatter(x[valid], y[valid], c=color_data[valid], cmap=cmap, norm=norm,
                        s=22, alpha=0.75, edgecolor="white", linewidth=0.4)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(color_label, fontsize=9)
    else:  # binary
        cd = color_data.astype(float)
        valid = ~np.isnan(cd) & ~np.isnan(x) & ~np.isnan(y)
        for v, c, lab in [(0, "#94a3b8", "No"), (1, "#dc2626", "Yes")]:
            m = valid & (cd == v)
            ax.scatter(x[m], y[m], c=c, s=22, alpha=0.75,
                       edgecolor="white", linewidth=0.4, label=f"{color_label}: {lab}")
        ax.legend(loc="best", fontsize=8, frameon=True)

    ax.axhline(0, color="grey", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel(label(x_name))
    ax.set_ylabel(f"SHAP value for {label(x_name)}\n(contribution to RSF risk score)")
    ttl = f"{label(x_name)} — SHAP dependence plot"
    if title_extra: ttl += f" · {title_extra}"
    ax.set_title(ttl)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    plt.savefig(FIG / fname)
    plt.close(fig)
    print(f"  {fname} saved")


# ---------------------------------------------------------------------
# Helper: 3-way faceted plot (continuous color, binary facet)
# ---------------------------------------------------------------------
def dep_facet(
    x_name, color_data, color_label, facet_data, facet_label,
    fname, cmap="viridis", color_vmin=None, color_vmax=None,
):
    if x_name not in fi:
        print(f"  [skip] feature '{x_name}' not in model"); return
    xi = fi[x_name]
    x = X[:, xi].astype(float)
    y = SV[:, xi]
    fd = facet_data.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    norm = Normalize(
        vmin=color_vmin if color_vmin is not None else np.nanpercentile(color_data, 5),
        vmax=color_vmax if color_vmax is not None else np.nanpercentile(color_data, 95),
    )
    for ax, (val, label_v) in zip(axes, [(0, f"{facet_label}: No"), (1, f"{facet_label}: Yes")]):
        m = ~np.isnan(fd) & (fd == val) & ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(color_data)
        sc = ax.scatter(x[m], y[m], c=color_data[m], cmap=cmap, norm=norm,
                        s=22, alpha=0.78, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="grey", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel(label(x_name))
        ax.set_title(f"{label_v}  (n = {m.sum()})")
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"SHAP value for {label(x_name)}")
    cbar = plt.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(color_label, fontsize=9)
    fig.suptitle(f"{label(x_name)} — SHAP dependence, faceted by {facet_label}",
                 fontsize=12, y=1.02)
    plt.savefig(FIG / fname)
    plt.close(fig)
    print(f"  {fname} saved")


# ---------------------------------------------------------------------
# Define plots (each generates one PNG)
# ---------------------------------------------------------------------
print("Generating dependence plots:")

# === 2-way continuous-color plots ===

# 1. Age vs SHAP, colored by post-SVA (clinically: does sagittal imbalance amplify age risk?)
dep_plot("Age", raw_col("post SVA"), "Post-op SVA (mm)",
         "fig16_dep_Age_by_postSVA.png", cmap="plasma",
         color_vmin=0, color_vmax=80)

# 2. Age vs SHAP, colored by levels fused proxy (multi-level interaction with age)
dep_plot("Age", levels_per_pt.astype(float), "Levels fused",
         "fig17_dep_Age_by_levelsFused.png", cmap="viridis",
         color_vmin=1, color_vmax=6)

# 3. Age vs SHAP, colored by ABS PI-LL mismatch (Schwab interaction)
dep_plot("Age", raw_col("ABS PI-LL angle mismatch"), "Absolute PI–LL mismatch (°)",
         "fig18_dep_Age_by_pillMismatch.png", cmap="magma",
         color_vmin=0, color_vmax=30)

# 4. PCA 1 (post-op alignment) vs SHAP, colored by Age
dep_plot("pca_num_1", X[:, fi["Age"]].astype(float), "Patient age (years)",
         "fig19_dep_PCA1_by_Age.png", cmap="viridis",
         color_vmin=40, color_vmax=80)

# 5. PCA 1 vs SHAP, colored by post-SVA (composite vs single alignment metric)
dep_plot("pca_num_1", raw_col("post SVA"), "Post-op SVA (mm)",
         "fig20_dep_PCA1_by_postSVA.png", cmap="plasma",
         color_vmin=0, color_vmax=80)

# 6. PCA 2 (operative burden) vs SHAP, colored by Age
dep_plot("pca_num_2", X[:, fi["Age"]].astype(float), "Patient age (years)",
         "fig21_dep_PCA2_by_Age.png", cmap="viridis",
         color_vmin=40, color_vmax=80)

# 7. PCA 2 vs SHAP, colored by levels fused
dep_plot("pca_num_2", levels_per_pt.astype(float), "Levels fused",
         "fig22_dep_PCA2_by_levelsFused.png", cmap="cividis",
         color_vmin=1, color_vmax=6)

# 8. PCA 3 (sagittal mismatch) vs SHAP, colored by post-SVA
dep_plot("pca_num_3", raw_col("post SVA"), "Post-op SVA (mm)",
         "fig23_dep_PCA3_by_postSVA.png", cmap="plasma",
         color_vmin=0, color_vmax=80)

# 9. PCA 3 vs SHAP, colored by Age
dep_plot("pca_num_3", X[:, fi["Age"]].astype(float), "Patient age (years)",
         "fig24_dep_PCA3_by_Age.png", cmap="viridis",
         color_vmin=40, color_vmax=80)

# === 2-way binary-color plots ===

# 10. Age vs SHAP, colored by PI>50 (binary)
dep_plot("Age",
         X[:, fi["(1 = PI>50)"]].astype(float),
         "PI > 50",
         "fig25_dep_Age_by_PIgt50.png",
         color_kind="binary")

# 11. Age vs SHAP, colored by prior surgery
dep_plot("Age",
         X[:, fi["prior back surgeries? (y=1)"]].astype(float),
         "Prior back surgery",
         "fig26_dep_Age_by_priorSurgery.png",
         color_kind="binary")

# 12. PCA 1 vs SHAP, colored by Anterior+Posterior approach
dep_plot("pca_num_1",
         X[:, fi["Anterior + Posterior Apporoach"]].astype(float),
         "A+P approach",
         "fig27_dep_PCA1_by_APapproach.png",
         color_kind="binary")

# === 3-way faceted plots ===

# 13. PCA 1 vs SHAP, colored by Age, faceted by PI>50
dep_facet("pca_num_1",
          X[:, fi["Age"]].astype(float), "Patient age",
          X[:, fi["(1 = PI>50)"]].astype(float), "PI > 50",
          "fig28_dep_PCA1_byAge_facetPIgt50.png",
          cmap="viridis", color_vmin=40, color_vmax=80)

# 14. Age vs SHAP, colored by post-SVA, faceted by prior surgery
dep_facet("Age",
          raw_col("post SVA"), "Post-op SVA (mm)",
          X[:, fi["prior back surgeries? (y=1)"]].astype(float), "Prior surgery",
          "fig29_dep_Age_byPostSVA_facetPriorSurg.png",
          cmap="plasma", color_vmin=0, color_vmax=80)

# 15. PCA 3 vs SHAP, colored by Age, faceted by A+P approach
dep_facet("pca_num_3",
          X[:, fi["Age"]].astype(float), "Patient age",
          X[:, fi["Anterior + Posterior Apporoach"]].astype(float), "A+P approach",
          "fig30_dep_PCA3_byAge_facetAPapproach.png",
          cmap="viridis", color_vmin=40, color_vmax=80)

print(f"\nAll dependence plots in {FIG}/")
print("\nFiles generated:")
for p in sorted(FIG.glob("fig1[6-9]*.png")) + sorted(FIG.glob("fig2*.png")) + sorted(FIG.glob("fig3*.png")):
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
