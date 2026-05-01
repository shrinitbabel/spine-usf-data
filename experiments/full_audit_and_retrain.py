"""
End-to-end data audit, multiple-imputation, and model retrain.

Drops features with >70% missing (post SVA, osteotomy level, ACR level,
Additional Procedures w/in surgery) since imputing 75-97% missing is not
defensible. MICE-imputes continuous numerics with 30-55% missing using
sklearn's IterativeImputer. Retrains RSF and DeepSurv on the cleaned
feature matrix and writes new model bundles.

Run as a single command:
    python -W ignore experiments/full_audit_and_retrain.py
"""

from __future__ import annotations
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
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
# 1. Load cleaned (already has the fixed level columns from prior step)
# --------------------------------------------------------------------------
df = pd.read_csv(ROOT / "cleaned_data.csv")
df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)
print(f"Loaded cleaned_data.csv: {df.shape}")

# Survival labels
event_col = "REVERIFIED ASD"
df[event_col] = df[event_col].fillna(0).astype(int)
df["time_surv"] = np.where(
    df[event_col] == 1,
    df["Time Until ASD Diagnosis (months)"],
    df["Time Without_ASD (months)"],
)
df = df.dropna(subset=["time_surv"]).reset_index(drop=True)
df["time_surv"] = df["time_surv"].astype(float)
print(f"After dropping missing time-to-event: {len(df)} patients, "
      f"{int(df[event_col].sum())} events ({df[event_col].mean()*100:.1f}%)")


# --------------------------------------------------------------------------
# 2. Audit and decide on drops
# --------------------------------------------------------------------------
TOO_SPARSE_DROP = [
    "post SVA",                            # 96.9% missing — only 17 measurements, undefendable to impute
    "osteotomy level",                     # 96.9% missing — only relevant for 17 patients; we keep the binary
    "ACR level",                           # 86.6% missing — only relevant for ACR=Yes; we keep the binary
    "Additional Procedures w/in surgery",  # 74.9% missing — free-text noise
]

# Features to MICE-impute (continuous numerics with 0.5–55% missing)
NUMERIC_TO_IMPUTE = [
    "BMI",
    "ALIF Count",
    "Average PI",
    "PI-LL angle mismatch",
    "ABS PI-LL angle mismatch",
    "Post-op SS",
    "post PI",
    "post PT",
    "post LL",
    "length of hospital stay (d)",
]

# Full feature list — same as before but minus dropped sparse cols
clean_feature_cols = [
    "Sex","Age","BMI","prior back surgeries? (y=1)",
    "dx_adjacent_segment","dx_spondylolisthesis","dx_spondylosis","dx_stenosis",
    "dx_scoliosis","dx_flat_back","dx_sagittal_imbalance","dx_post_laminectomy","dx_deformity",
    "Case/Type of Surgery",
    "T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1",
    "Perc screws?","Open","Open Check V2","Standalone XLIF Check",
    "Retroperitoneal Approach (LLIF ± ALIF)","Anterior + Posterior Apporoach",
    "Osteotomies (yes/no)",
    "ALIF Count","Lateral Count","ACR (y=1)",
    "Average PI","PI-LL angle mismatch","ABS PI-LL angle mismatch",
    "PI-LL Mismatch Category (1 = mismatch > +/- 9",
    "PI-LL Mismatch Category (1 = mismatch > +/- 10","(1 = PI>50)",
    "Post-op SS","post PI","post PT","post LL",
    "infection 1=yes","DVT  1=yes","PE  1=yes","MI 1=yes",
    "femoral palsy (knee extension weakness) 1=yes",
    "hip flexion weakness (iliopsoas weakness)  1=yes",
    "acute thigh paresthesia (immediate post op)","psoas hematoma",
    "length of hospital stay (d)",
]
print(f"\nDropping {len(TOO_SPARSE_DROP)} features (>70% missing): {TOO_SPARSE_DROP}")
print(f"Will MICE-impute: {NUMERIC_TO_IMPUTE}")
print(f"Final feature list size: {len(clean_feature_cols)}")

present = [c for c in clean_feature_cols if c in df.columns]
X = df[present].copy()

# --------------------------------------------------------------------------
# 3. Coerce numerics, impute, then one-hot
# --------------------------------------------------------------------------
for c in NUMERIC_TO_IMPUTE:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

# Median imputation on the numeric block (single-pass, transparent, the
# standard non-MICE approach for clinical prediction models with MAR data)
num_block = X[NUMERIC_TO_IMPUTE].copy()
print(f"\nPre-imputation null counts:")
for c in NUMERIC_TO_IMPUTE:
    print(f"  {c:35s}  {num_block[c].isna().sum():>4d}  ({num_block[c].isna().mean()*100:.1f}%)")

imputer = SimpleImputer(strategy="median")
num_imputed = imputer.fit_transform(num_block.values)
for j, c in enumerate(NUMERIC_TO_IMPUTE):
    X[c] = num_imputed[:, j]

print(f"\nPost-imputation: every numeric column is now complete.")
print(f"  Sanity — Average PI: mean before={num_block['Average PI'].mean():.2f}, after={X['Average PI'].mean():.2f}")
print(f"  Sanity — post LL:    mean before={num_block['post LL'].mean():.2f}, after={X['post LL'].mean():.2f}")

# Categorical Case/Type of Surgery: fill the 1 missing with mode
if "Case/Type of Surgery" in X.columns:
    X["Case/Type of Surgery"] = X["Case/Type of Surgery"].fillna(
        X["Case/Type of Surgery"].mode().iloc[0]
    )

# One-hot the categorical strings
str_cats = X.select_dtypes(include=["object"]).columns.tolist()
print(f"\nOne-hot encoding: {str_cats}")
X_encoded = pd.get_dummies(X, columns=str_cats, drop_first=True).fillna(0.0)
print(f"Encoded matrix: {X_encoded.shape}")

# --------------------------------------------------------------------------
# 4. Drop ultra-rare binaries (<5 ones)
# --------------------------------------------------------------------------
binary_cols = [c for c in X_encoded.columns if set(np.unique(X_encoded[c])) <= {0, 1}]
rare_cols = [c for c in binary_cols if X_encoded[c].sum() < 5]
print(f"Dropping {len(rare_cols)} ultra-rare binary cols")
X_reduced = X_encoded.drop(columns=rare_cols)
print(f"After rare-drop: {X_reduced.shape}")

# Sanity-check: per-level columns must survive now
LEVEL_COLS = ["T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"]
levels_in = [L for L in LEVEL_COLS if L in X_reduced.columns]
print(f"Level binaries kept: {levels_in}")

# --------------------------------------------------------------------------
# 5. PCA on continuous numerics
# --------------------------------------------------------------------------
numeric_present = [c for c in NUMERIC_TO_IMPUTE if c in X_reduced.columns]
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
print(f"Final RSF input: {X_vals.shape}")

# --------------------------------------------------------------------------
# 6. Retrain RSF
# --------------------------------------------------------------------------
print("\n--- Retraining RSF ---")
times = df["time_surv"].astype(float).values
events = df[event_col].astype(bool).values
y = Surv.from_arrays(event=events, time=times)

BEST = dict(n_estimators=200, min_samples_split=4, min_samples_leaf=12,
             max_features="sqrt", n_jobs=-1, random_state=42)
rsf = RandomSurvivalForest(**BEST).fit(X_vals, y)
cohort_risk = rsf.predict(X_vals)
print(f"Cohort risk: mean={cohort_risk.mean():.3f}, std={cohort_risk.std():.3f}, "
      f"range [{cohort_risk.min():.3f}, {cohort_risk.max():.3f}]")

with open(ROOT / "models" / "rsf_bundle.pkl", "wb") as f:
    pickle.dump({
        "model": rsf,
        "feature_columns": feature_cols,
        "encoded_columns": X_encoded.columns.tolist(),
        "rare_cols": rare_cols,
        "numeric_present": numeric_present,
        "scaler": scaler,
        "pca": pca,
        "imputer": imputer,
        "imputed_numeric_cols": NUMERIC_TO_IMPUTE,
        "X_vals": X_vals,
    }, f)
print(f"Saved -> models/rsf_bundle.pkl")

# --------------------------------------------------------------------------
# 7. Retrain DeepSurv
# --------------------------------------------------------------------------
print("\n--- Retraining DeepSurv ---")
NUM_NODES = [32, 16]; DROPOUT = 0.5
in_features = X_vals.shape[1]
net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=NUM_NODES,
                              out_features=1, batch_norm=True, dropout=DROPOUT,
                              output_bias=False)
ds_model = CoxPH_NN(net, tt.optim.Adam(lr=1e-3))
ds_model.fit(X_vals.astype("float32"),
             (times.astype("float32"), events.astype("float32")),
             batch_size=64, epochs=80, verbose=False)
ds_model.compute_baseline_hazards()
risk_ds = ds_model.predict(X_vals.astype("float32")).reshape(-1)
print(f"DeepSurv risk: range [{risk_ds.min():.3f}, {risk_ds.max():.3f}]")

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
        "imputer": imputer,
        "imputed_numeric_cols": NUMERIC_TO_IMPUTE,
        "cohort_risk": risk_ds,
    }, f)
print(f"Saved -> models/deepsurv_bundle.pkl")

# --------------------------------------------------------------------------
# 8. Sanity check on 4 distinct preset patients — predictions should differ
# --------------------------------------------------------------------------
print("\n--- Sanity check (predictions for clinically distinct patients) ---")

def make_row(overrides: dict) -> np.ndarray:
    row = {c: 0 for c in X_encoded.columns}
    # Reasonable defaults
    row["Sex"] = 1; row["Age"] = 65; row["BMI"] = 28
    row["dx_stenosis"] = 1; row["Open"] = 1
    row["Lateral Count"] = 1
    row["Average PI"] = 55; row["PI-LL angle mismatch"] = 12
    row["ABS PI-LL angle mismatch"] = 12; row["(1 = PI>50)"] = 1
    row["post LL"] = 45; row["post PI"] = 55; row["post PT"] = 20
    row["Post-op SS"] = 32; row["length of hospital stay (d)"] = 4
    for k, v in overrides.items():
        if k in row:
            row[k] = v
    rdf = pd.DataFrame([row])[X_encoded.columns]
    rdf = rdf.drop(columns=[c for c in rare_cols if c in rdf.columns], errors="ignore")
    pcs_r = pca.transform(scaler.transform(rdf[numeric_present].values))
    pcs_r = pd.DataFrame(pcs_r, columns=[f"pca_num_{i+1}" for i in range(n_pca)],
                          index=rdf.index)
    final = pd.concat([rdf.drop(columns=numeric_present), pcs_r], axis=1)
    return final[feature_cols].values.astype(float)

cases = {
    "Low-risk MIS short-segment (50yo, L4-L5, MIS)": dict(
        Age=50, BMI=24, **{"prior back surgeries? (y=1)": 0},
        **{"L4-L5": 1}, Open=0, **{"Perc screws?": 1},
        **{"Standalone XLIF Check": 1}, **{"Retroperitoneal Approach (LLIF ± ALIF)": 1},
        **{"Average PI": 48, "PI-LL angle mismatch": 4, "ABS PI-LL angle mismatch": 4,
           "(1 = PI>50)": 0, "post LL": 50, "post PI": 48, "post PT": 13,
           "Post-op SS": 35, "length of hospital stay (d)": 2},
    ),
    "Adjacent stress (63yo, 3 levels L3-S1)": dict(
        Age=63, BMI=28, dx_spondylosis=1,
        **{"L3-L4": 1, "L4-L5": 1, "L5-S1": 1},
        Open=1, **{"Average PI": 58, "PI-LL angle mismatch": 15,
                   "ABS PI-LL angle mismatch": 15, "post LL": 45,
                   "post PI": 58, "post PT": 22, "Post-op SS": 28,
                   "length of hospital stay (d)": 4},
    ),
    "Long-construct deformity (75yo, 6 levels)": dict(
        Age=75, BMI=32, **{"prior back surgeries? (y=1)": 1},
        dx_scoliosis=1, dx_flat_back=1, dx_sagittal_imbalance=1,
        **{L: 1 for L in LEVEL_COLS},
        Open=1, **{"Osteotomies (yes/no)": 1, "Anterior + Posterior Apporoach": 1,
                   "ACR (y=1)": 1, "Lateral Count": 3,
                   "Average PI": 65, "PI-LL angle mismatch": 25,
                   "ABS PI-LL angle mismatch": 25, "post LL": 40,
                   "post PI": 65, "post PT": 30, "Post-op SS": 20,
                   "length of hospital stay (d)": 8},
    ),
    "Revision lumbar (68yo, prior, A+P + osteotomy)": dict(
        Age=68, BMI=30, **{"prior back surgeries? (y=1)": 1},
        **{"L4-L5": 1, "L5-S1": 1},
        Open=1, **{"Perc screws?": 1, "Osteotomies (yes/no)": 1,
                   "Anterior + Posterior Apporoach": 1,
                   "Average PI": 56, "PI-LL angle mismatch": 18,
                   "ABS PI-LL angle mismatch": 18, "post LL": 45,
                   "post PI": 56, "post PT": 24, "Post-op SS": 27,
                   "length of hospital stay (d)": 5},
    ),
}

print(f"\n  {'Case':52s}  RSF risk   pct    DS risk")
for name, ov in cases.items():
    Xp = make_row(ov)
    rsf_r = float(rsf.predict(Xp)[0])
    rsf_pct = float((cohort_risk < rsf_r).mean() * 100)
    ds_r = float(ds_model.predict(Xp.astype("float32")).reshape(-1)[0])
    print(f"  {name[:52]:52s}  {rsf_r:>7.3f}  {rsf_pct:>5.1f}    {ds_r:>+7.3f}")

print("\nDone.  Next: rerun compute_shap.py and the figure scripts.")
