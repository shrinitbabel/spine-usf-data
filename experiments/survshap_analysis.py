"""
Time-dependent SHAP (survSHAP(t)) analysis of the deployed RSF.

Per-patient checkpointing: after each patient, the cumulative SHAP tensor is
re-saved to experiments/survshap_checkpoint.npz.  Re-running this script will
resume from the last completed patient.

Generates (after all patients done):
  fig32_survshap_global_over_time.png
  fig33_survshap_temporal_top5.png
  fig34_survshap_individual_cases.png
  fig35_survshap_vs_static_shap.png
"""
from __future__ import annotations
import sys, os, time, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# Force unbuffered stdout so progress is visible
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)
CKPT = ROOT / "experiments" / "survshap_checkpoint.npz"

print(f"[{time.strftime('%H:%M:%S')}] loading bundle...", flush=True)
with open(ROOT / "models" / "rsf_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

rsf = bundle["model"]; X = bundle["X_vals"]; fc = bundle["feature_columns"]
print(f"[{time.strftime('%H:%M:%S')}] RSF: {len(fc)} features, {X.shape[0]} patients", flush=True)

df = pd.read_csv(ROOT / "cleaned_data.csv")
df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)
event_col = "REVERIFIED ASD"
df[event_col] = df[event_col].fillna(0).astype(int)
df["time_surv"] = np.where(df[event_col]==1,
                           df["Time Until ASD Diagnosis (months)"],
                           df["Time Without_ASD (months)"])
df = df.dropna(subset=["time_surv"]).reset_index(drop=True)
times = df["time_surv"].astype(float).values
events = df[event_col].astype(bool).values
assert len(times) == X.shape[0]

from sksurv.util import Surv
y = Surv.from_arrays(event=events, time=times)
X_df = pd.DataFrame(X, columns=fc)

# --------------------------------------------------------------------------
# Pick the sample (deterministic, so resume picks the same patients)
# --------------------------------------------------------------------------
N_SAMPLE = 60
HORIZONS = np.array([12., 24., 36., 48., 60.])
rng = np.random.default_rng(42)
ev_i = np.where(events)[0]; nev_i = np.where(~events)[0]
sample_idx = np.concatenate([
    rng.choice(ev_i, size=min(30, len(ev_i)), replace=False),
    rng.choice(nev_i, size=min(30, len(nev_i)), replace=False),
])
print(f"[{time.strftime('%H:%M:%S')}] sample: n={len(sample_idx)} ({events[sample_idx].sum()} events)", flush=True)

# Load checkpoint or init
if CKPT.exists():
    ck = np.load(CKPT, allow_pickle=True)
    shap_tensor = ck["shap_tensor"]
    done_mask = ck["done_mask"]
    print(f"[{time.strftime('%H:%M:%S')}] resumed checkpoint: {int(done_mask.sum())}/{len(sample_idx)} patients done", flush=True)
else:
    shap_tensor = np.zeros((len(sample_idx), len(fc), len(HORIZONS)), dtype=np.float32)
    done_mask = np.zeros(len(sample_idx), dtype=bool)

# --------------------------------------------------------------------------
# Per-patient SHAP via PredictSurvSHAP (sampling)
# --------------------------------------------------------------------------
from survshap import SurvivalModelExplainer, PredictSurvSHAP

print(f"[{time.strftime('%H:%M:%S')}] building explainer...", flush=True)
explainer = SurvivalModelExplainer(model=rsf, data=X_df, y=y)

t0 = time.time()
for i, gi in enumerate(sample_idx):
    if done_mask[i]:
        continue
    pat_t0 = time.time()
    try:
        ps = PredictSurvSHAP(function_type="sf", calculation_method="sampling",
                             aggregation_method="integral", B=25, random_state=42)
        ps.fit(explainer, X_df.iloc[[gi]], timestamps=HORIZONS)
        res = ps.result
        feat_col = "variable_name" if "variable_name" in res.columns else "variable"
        t_cols = [c for c in res.columns if c.startswith("t = ")]
        t_vals = np.array([float(c.replace("t = ","")) for c in t_cols])
        nearest = [t_cols[int(np.argmin(np.abs(t_vals - h)))] for h in HORIZONS]
        for j, fname in enumerate(fc):
            row = res[res[feat_col] == fname]
            if len(row):
                shap_tensor[i, j, :] = np.abs(row[nearest].values[0].astype(float))
        done_mask[i] = True
        np.savez(CKPT, shap_tensor=shap_tensor, done_mask=done_mask,
                 sample_idx=sample_idx, horizons=HORIZONS,
                 feature_cols=np.array(fc, dtype=object))
        elapsed = time.time() - t0
        per = elapsed / max(1, int(done_mask.sum()))
        remain = per * (len(sample_idx) - int(done_mask.sum()))
        print(f"[{time.strftime('%H:%M:%S')}] patient {i+1}/{len(sample_idx)} done in "
              f"{time.time()-pat_t0:.1f}s | total {elapsed/60:.1f}m | ETA {remain/60:.1f}m", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] patient {i+1} FAILED: {type(e).__name__}: {e}", flush=True)
        continue

print(f"[{time.strftime('%H:%M:%S')}] all patients done. building figures...", flush=True)

# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------
mean_abs_t = shap_tensor[done_mask].mean(axis=0)
imp_df = pd.DataFrame(mean_abs_t, index=fc, columns=[f"{int(h)}mo" for h in HORIZONS])
imp_df["mean"] = imp_df.mean(axis=1)
imp_df = imp_df.sort_values("mean", ascending=False)
imp_df.to_csv(ROOT / "experiments" / "survshap_global_importance.csv")
print("\nTop 15 features by time-averaged |SHAP|:")
print(imp_df.head(15).round(4).to_string(), flush=True)

# fig32: heatmap
TOP = 12
top_feats = imp_df.head(TOP).index.tolist()
H = imp_df.loc[top_feats, [f"{int(h)}mo" for h in HORIZONS]].values
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(H, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels([f"{int(h)} mo" for h in HORIZONS])
ax.set_yticks(range(TOP)); ax.set_yticklabels([f.replace("Additional Procedures w/in surgery_","ap_")[:42] for f in top_feats])
ax.set_xlabel("Follow-up horizon")
ax.set_title("Time-dependent feature importance (mean |survSHAP(t)|)")
fig.colorbar(im, ax=ax, label="mean |SHAP|"); fig.tight_layout()
fig.savefig(FIGS / "fig32_survshap_global_over_time.png", dpi=300); plt.close(fig)

# fig33: line plot of top-5
TOP5 = imp_df.head(5).index.tolist()
fig, ax = plt.subplots(figsize=(8, 5))
for f in TOP5:
    ax.plot(HORIZONS, imp_df.loc[f, [f"{int(h)}mo" for h in HORIZONS]].values,
            marker="o", linewidth=2, label=f[:38])
ax.set_xlabel("Follow-up horizon (months)"); ax.set_ylabel("mean |SHAP|")
ax.set_title("Temporal evolution of top-5 risk drivers")
ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
fig.savefig(FIGS / "fig33_survshap_temporal_top5.png", dpi=300); plt.close(fig)

# fig35: static vs time-averaged comparison (skip fig34 individual case for now -
#        requires re-running PredictSurvSHAP per case which we already have via checkpoint)
import shap
bg = rng.choice(len(X), 50, replace=False)
sm = rng.choice(len(X), 120, replace=False)
exp_static = shap.PermutationExplainer(rsf.predict, X[bg])
sv_static = exp_static(X[sm], max_evals=400).values
static_imp = pd.Series(np.abs(sv_static).mean(axis=0), index=fc).sort_values(ascending=False)
merged = pd.concat([static_imp.head(15).rename("static_SHAP"),
                    imp_df["mean"].head(15).rename("survSHAP_avg")], axis=1).fillna(0).head(20)
fig, ax = plt.subplots(figsize=(9, 7))
yp = np.arange(len(merged))
ax.barh(yp - 0.2, merged["static_SHAP"], 0.4, label="static SHAP", color="#888")
ax.barh(yp + 0.2, merged["survSHAP_avg"], 0.4, label="survSHAP(t) (time-avg)", color="#3a7")
ax.set_yticks(yp); ax.set_yticklabels([f[:40] for f in merged.index], fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("mean |SHAP|"); ax.legend()
ax.set_title("Static vs time-dependent SHAP ranking"); fig.tight_layout()
fig.savefig(FIGS / "fig35_survshap_vs_static_shap.png", dpi=300); plt.close(fig)

print(f"[{time.strftime('%H:%M:%S')}] all figures written to figures/", flush=True)
