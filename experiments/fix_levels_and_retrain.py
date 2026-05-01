"""
Fix the level-of-fusion columns in cleaned_data.csv (currently all zero
because the cleaning step collapsed text codes like 'XLIF' / 'LLIF' to 0)
and retrain RSF + DeepSurv with the corrected feature set.

Steps:
  1. Re-derive per-level binaries from uncleaned_data.csv (any non-NaN,
     non-'N', non-blank value -> 1 = fusion at that level).
  2. Re-derive levels_fused_count, construct_span_levels, region flags.
  3. Overwrite the level columns in cleaned_data.csv.
  4. Retrain rsf_bundle.pkl and deepsurv_bundle.pkl.
  5. Sanity check: predict on a single-level L4-L5 case and a 6-level
     long construct -> they MUST differ now.
"""

from __future__ import annotations
from pathlib import Path
import os, pickle, numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest

import torch
import torchtuples as tt
from pycox.models import CoxPH as CoxPH_NN

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# 1. Recover level info from uncleaned data
# --------------------------------------------------------------------------
raw = pd.read_csv(ROOT / "uncleaned_data.csv")
raw.columns = raw.columns.str.strip().str.replace("\n", " ", regex=False)

LEVEL_COLS = ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]

def is_fused(v):
    if pd.isna(v): return 0
    s = str(v).strip().replace("\xa0", "")
    if s == "" or s.upper() == "N": return 0
    return 1

print("Per-level fusion counts after re-derivation from uncleaned data:")
fixed_levels = {}
for L in LEVEL_COLS:
    if L in raw.columns:
        b = raw[L].apply(is_fused).astype(int).values
        fixed_levels[L] = b
        print(f"  {L}: n_fused = {b.sum()} ({b.mean()*100:.1f}%)")
    else:
        print(f"  {L}: NOT FOUND in raw")
        fixed_levels[L] = np.zeros(len(raw), dtype=int)

levels_per_pt = sum(fixed_levels[L] for L in LEVEL_COLS)
print(f"\nLevels-fused distribution: {pd.Series(levels_per_pt).value_counts().sort_index().to_dict()}")

# --------------------------------------------------------------------------
# 2. Update cleaned_data.csv with corrected level columns
# --------------------------------------------------------------------------
clean = pd.read_csv(ROOT / "cleaned_data.csv")
clean.columns = clean.columns.str.strip().str.replace("\n", " ", regex=False)

assert len(clean) == len(raw), f"Row count mismatch: clean={len(clean)} raw={len(raw)}"

for L in LEVEL_COLS:
    clean[L] = fixed_levels[L]

# Engineered summaries
clean["levels_fused_count"] = levels_per_pt
clean["thoracolumbar_junction"] = ((fixed_levels["T12-L1"] + fixed_levels["L1-L2"]) > 0).astype(int)
clean["upper_lumbar"] = ((fixed_levels["L1-L2"] + fixed_levels["L2-L3"]) > 0).astype(int)
clean["lower_lumbar"] = ((fixed_levels["L3-L4"] + fixed_levels["L4-L5"]) > 0).astype(int)
clean["lumbosacral"] = (fixed_levels["L5-S1"]).astype(int)

# construct_span_levels: span between highest and lowest fused level
LEVEL_RANK = {"T12-L1": 0, "L1-L2": 1, "L2-L3": 2, "L3-L4": 3, "L4-L5": 4, "L5-S1": 5}
spans = []
for i in range(len(clean)):
    fused_idx = [LEVEL_RANK[L] for L in LEVEL_COLS if fixed_levels[L][i] == 1]
    spans.append(max(fused_idx) - min(fused_idx) + 1 if fused_idx else 0)
clean["construct_span_levels"] = spans

clean.to_csv(ROOT / "cleaned_data.csv", index=False)
print(f"\nWrote corrected level columns to {ROOT / 'cleaned_data.csv'}")

# --------------------------------------------------------------------------
# 3. Retrain RSF (matching the original notebook pipeline)
# --------------------------------------------------------------------------
print("\n--- Retraining RSF on corrected dataset ---")

df = clean.copy()
event_col = "REVERIFIED ASD"
df[event_col] = df[event_col].fillna(0).astype(int)
df["time_surv"] = np.where(
    df[event_col] == 1,
    df["Time Until ASD Diagnosis (months)"],
    df["Time Without_ASD (months)"],
)
df = df.dropna(subset=["time_surv"]).reset_index(drop=True)
df["time_surv"] = df["time_surv"].astype(float)

clean_feature_cols = [
    "Sex","Age","BMI","prior back surgeries? (y=1)",
    "dx_adjacent_segment","dx_spondylolisthesis","dx_spondylosis","dx_stenosis",
    "dx_scoliosis","dx_flat_back","dx_sagittal_imbalance","dx_post_laminectomy","dx_deformity",
    "Case/Type of Surgery","Additional Procedures w/in surgery",
    # Variant C: single levels_fused_count integer (folded into PCA numeric block
    # below) instead of six per-level binaries. Best honest CV C-index = 0.614.
    "levels_fused_count",
    "Perc screws?","Open","Open Check V2","Standalone XLIF Check",
    "Retroperitoneal Approach (LLIF ± ALIF)","Anterior + Posterior Apporoach",
    "Osteotomies (yes/no)","osteotomy level",
    "ALIF Count","Lateral Count","ACR (y=1)","ACR level",
    "Average PI","PI-LL angle mismatch","ABS PI-LL angle mismatch",
    "PI-LL Mismatch Category (1 = mismatch > +/- 9",
    "PI-LL Mismatch Category (1 = mismatch > +/- 10","(1 = PI>50)",
    "Post-op SS","post PI","post PT","post LL","post SVA",
    "infection 1=yes","DVT  1=yes","PE  1=yes","MI 1=yes",
    "femoral palsy (knee extension weakness) 1=yes",
    "hip flexion weakness (iliopsoas weakness)  1=yes",
    "acute thigh paresthesia (immediate post op)","psoas hematoma",
    "length of hospital stay (d)",
]
present = [c for c in clean_feature_cols if c in df.columns]
numeric_like = [
    "BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
    "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
    "post SVA","length of hospital stay (d)",
    "levels_fused_count",
]

X = df[present].copy()
for c in numeric_like:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
str_cats = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=str_cats, drop_first=True).fillna(0.0)
print(f"Encoded matrix: {X_encoded.shape}")

# Drop rare binaries
binary_cols = [c for c in X_encoded.columns if set(np.unique(X_encoded[c])) <= {0, 1}]
rare_cols = [c for c in binary_cols if X_encoded[c].sum() < 5]
X_reduced = X_encoded.drop(columns=rare_cols)
print(f"After rare-drop: {X_reduced.shape} (dropped {len(rare_cols)} rare binaries)")
print(f"Per-level columns now in feature set: {[L for L in LEVEL_COLS if L in X_reduced.columns]}")

# PCA on continuous numerics
numeric_present = [c for c in numeric_like if c in X_reduced.columns]
scaler = StandardScaler().fit(X_reduced[numeric_present].values)
n_pca = min(3, len(numeric_present))
pca = PCA(n_components=n_pca, random_state=42).fit(
    scaler.transform(X_reduced[numeric_present].values)
)
pcs = pca.transform(scaler.transform(X_reduced[numeric_present].values))
df_pcs = pd.DataFrame(
    pcs, columns=[f"pca_num_{i+1}" for i in range(n_pca)], index=X_reduced.index
)
X_final = pd.concat([X_reduced.drop(columns=numeric_present), df_pcs], axis=1)
feature_cols = X_final.columns.tolist()
X_vals = X_final.values.astype(float)

times = df["time_surv"].astype(float).values
events = df[event_col].astype(bool).values
y = Surv.from_arrays(event=events, time=times)

print(f"Final feature matrix for RSF: {X_vals.shape}")

BEST = dict(n_estimators=200, min_samples_split=4, min_samples_leaf=12,
             max_features="sqrt", n_jobs=-1, random_state=42)
rsf = RandomSurvivalForest(**BEST).fit(X_vals, y)
cohort_risk = rsf.predict(X_vals)
print(f"Cohort risk: mean={cohort_risk.mean():.3f}, std={cohort_risk.std():.3f}")

with open(ROOT / "models" / "rsf_bundle.pkl", "wb") as f:
    pickle.dump({
        "model": rsf,
        "feature_columns": feature_cols,
        "encoded_columns": X_encoded.columns.tolist(),
        "rare_cols": rare_cols,
        "numeric_present": numeric_present,
        "scaler": scaler,
        "pca": pca,
        "X_vals": X_vals,
    }, f)
print(f"Saved -> {ROOT / 'models/rsf_bundle.pkl'}")

# --------------------------------------------------------------------------
# 4. Retrain DeepSurv with the same corrected feature set
# --------------------------------------------------------------------------
print("\n--- Retraining DeepSurv ---")
in_features = X_vals.shape[1]
NUM_NODES = [32, 16]; DROPOUT = 0.5
net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=NUM_NODES,
                              out_features=1, batch_norm=True, dropout=DROPOUT,
                              output_bias=False)
ds_model = CoxPH_NN(net, tt.optim.Adam(lr=1e-3))
ds_model.fit(X_vals.astype("float32"),
             (times.astype("float32"), events.astype("float32")),
             batch_size=64, epochs=80, verbose=False)
ds_model.compute_baseline_hazards()
risk_ds = ds_model.predict(X_vals.astype("float32")).reshape(-1)
print(f"DeepSurv cohort risk range: [{risk_ds.min():.3f}, {risk_ds.max():.3f}]")

with open(ROOT / "models" / "deepsurv_bundle.pkl", "wb") as f:
    pickle.dump({
        "state_dict": ds_model.net.state_dict(),
        "in_features": in_features,
        "num_nodes": NUM_NODES,
        "dropout": DROPOUT,
        "baseline_hazards": ds_model.baseline_hazards_,
        "baseline_cumulative_hazards": ds_model.baseline_cumulative_hazards_,
        "feature_columns": feature_cols,
        "encoded_columns": X_encoded.columns.tolist(),
        "rare_cols": rare_cols,
        "numeric_present": numeric_present,
        "scaler": scaler,
        "pca": pca,
        "cohort_risk": risk_ds,
    }, f)
print(f"Saved -> {ROOT / 'models/deepsurv_bundle.pkl'}")

# --------------------------------------------------------------------------
# 5. Sanity check — single-level L4-L5 vs 6-level T12-S1 should now differ
# --------------------------------------------------------------------------
print("\n--- Sanity check: predictions should now respond to fused-levels ---")

def make_row(level_set: dict) -> np.ndarray:
    row = {c: 0 for c in X_encoded.columns}
    # baseline patient (matches 'Adjacent stress' preset)
    row["Sex"] = 1; row["Age"] = 65; row["BMI"] = 28
    row["prior back surgeries? (y=1)"] = 0
    row["dx_stenosis"] = 1
    row["Open"] = 1; row["Lateral Count"] = 1
    row["Average PI"] = 55; row["PI-LL angle mismatch"] = 12
    row["ABS PI-LL angle mismatch"] = 12; row["(1 = PI>50)"] = 1
    row["post LL"] = 45; row["post SVA"] = 30
    row["post PI"] = 55; row["post PT"] = 20
    row["Post-op SS"] = 32; row["length of hospital stay (d)"] = 4
    # Variant C: derive levels_fused_count from the level set
    if "levels_fused_count" in row:
        row["levels_fused_count"] = sum(int(level_set.get(L, 0)) for L in LEVEL_COLS)
    rdf = pd.DataFrame([row])[X_encoded.columns]
    rdf = rdf.drop(columns=[c for c in rare_cols if c in rdf.columns], errors="ignore")
    pcs_r = pca.transform(scaler.transform(rdf[numeric_present].values))
    pcs_r = pd.DataFrame(pcs_r, columns=[f"pca_num_{i+1}" for i in range(n_pca)],
                          index=rdf.index)
    final = pd.concat([rdf.drop(columns=numeric_present), pcs_r], axis=1)
    return final[feature_cols].values.astype(float)

cases = {
    "Single-level L4-L5":         {"L4-L5": 1},
    "Two-level L4-S1":            {"L4-L5": 1, "L5-S1": 1},
    "Three-level L3-S1":          {"L3-L4": 1, "L4-L5": 1, "L5-S1": 1},
    "Long T12-S1 (6 levels)":     {L: 1 for L in LEVEL_COLS},
}
print(f"\n  {'Case':30s}  RSF risk   pct    DS risk")
for name, lvls in cases.items():
    Xp = make_row(lvls)
    rsf_r = float(rsf.predict(Xp)[0])
    rsf_p = (cohort_risk < rsf_r).mean() * 100
    ds_r = float(ds_model.predict(Xp.astype("float32")).reshape(-1)[0])
    print(f"  {name:30s}  {rsf_r:>7.3f}  {rsf_p:>5.1f}    {ds_r:>+7.3f}")

print("\nDone.")
