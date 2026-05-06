"""
Publication-style SHAP summary using clinical labels and a violin
(sina-strip) layout instead of the default beeswarm.

Replaces fig14 with a manuscript-ready version.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LinearSegmentedColormap

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

# Clinical, plain-language labels for each engineered feature.
PRETTY = {
    "pca_num_1": "PCA 1",
    "pca_num_2": "PCA 2",
    "pca_num_3": "PCA 3",
    "Age": "Patient age",
    "(1 = PI>50)": "High pelvic incidence (PI > 50°)",
    "Anterior + Posterior Apporoach": "Combined anterior + posterior approach",
    "Perc screws?": "Percutaneous pedicle-screw instrumentation",
    "Standalone XLIF Check": "Standalone lateral interbody fusion",
    "Open Check V2": "Open posterior approach",
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
    "dx_adjacent_segment": "Pre-op diagnosis: adjacent segment disease",
    "Additional Procedures w/in surgery_perc screws":
        "Additional intra-operative procedures",
}
def pretty(name: str) -> str:
    return PRETTY.get(name, name)


# Load saved cohort SHAP
d = np.load(EXP / "shap_values_cohort.npz", allow_pickle=True)
SV = d["shap_values"]
X = np.asarray(d["X_vals"], dtype=float)
feats = list(d["feature_columns"])
N, P = SV.shape

mean_abs = np.abs(SV).mean(axis=0)
top_n = 15
top_idx = np.argsort(mean_abs)[-top_n:]   # ascending so largest is at top of horizontal axis
labels = [pretty(feats[i]) for i in top_idx]

# Normalize each feature's value to [0, 1] for color mapping using percentile
# clipping (more robust to outliers than min/max)
def norm01(v):
    v = v.astype(float)
    if np.unique(v[~np.isnan(v)]).size <= 2:
        # binary-ish — map 0 to 0 and 1 to 1
        out = (v - np.nanmin(v)) / max(np.nanmax(v) - np.nanmin(v), 1e-9)
        return np.clip(out, 0, 1)
    lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
    if hi <= lo:
        return np.full_like(v, 0.5)
    return np.clip((v - lo) / (hi - lo), 0, 1)


# Custom colormap matching the SHAP convention (low = blue, high = red)
shap_cmap = LinearSegmentedColormap.from_list(
    "shap_div", ["#1e77b4", "#a78dc3", "#dc2626"], N=256
)

# Sina-style strip-violin: plot all points with x = SHAP value, y = jittered
# row position; colored by normalized feature value.
fig, ax = plt.subplots(figsize=(10.5, 8.5))

rng = np.random.RandomState(42)
for row, fi in enumerate(top_idx):
    sv_i = SV[:, fi]
    # Make jitter density-aware: more crowded near zero gets less spread
    kde_jitter = rng.normal(0, 0.16, size=N)
    # KDE-ish density via quick histogram
    counts, edges = np.histogram(sv_i, bins=30)
    bin_idx = np.clip(np.digitize(sv_i, edges) - 1, 0, len(counts) - 1)
    density = counts[bin_idx]
    density = density / max(density.max(), 1)
    jitter = kde_jitter * density   # taller stacks where density is high
    y = np.full(N, row, dtype=float) + jitter

    color_v = norm01(X[:, fi])
    sc = ax.scatter(sv_i, y, c=color_v, cmap=shap_cmap, s=18,
                    alpha=0.8, edgecolor="white", linewidth=0.3,
                    vmin=0, vmax=1)

ax.axvline(0, color="grey", linestyle=":", linewidth=1, alpha=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(labels)
ax.set_xlabel("SHAP value\n(contribution to predicted ASD risk, integrated over the full follow-up window)")
ax.set_title(
    "Per-patient SHAP contributions to overall ASD risk across follow-up  (top 15 features)",
    pad=14,
)
ax.grid(axis="x", alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
ax.set_ylim(-0.7, top_n - 0.3)

# Discrete colorbar with low / mid / high tick labels
cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(["Low", "Mid", "High"])
cbar.set_label("Feature value", rotation=270, labelpad=14)

# Soft annotation arrows to make the direction unambiguous
ax.annotate("Lower predicted risk", xy=(-1.5, top_n - 0.6), xytext=(-1.5, top_n - 0.6),
            ha="center", fontsize=9, color="#1e77b4", fontweight="bold")
ax.annotate("Higher predicted risk", xy=(1.5, top_n - 0.6), xytext=(1.5, top_n - 0.6),
            ha="center", fontsize=9, color="#dc2626", fontweight="bold")

plt.savefig(FIG / "fig14_shap_summary.png")
plt.close(fig)
print("fig14 (publication-style violin) saved")
