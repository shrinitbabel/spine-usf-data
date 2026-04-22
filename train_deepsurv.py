"""
Train DeepSurv on the full cohort and save a serving bundle.

Output: models/deepsurv_bundle.pkl  — drop-in companion to rsf_bundle.pkl.
Used by app.py's /predict/asd/deepsurv endpoint.

DeepSurv has the best calibration of the models we tested (ECE = 0.039 at
12mo, 0.052 at 24mo). Slightly worse discrimination than RSF, but
clinically more honest probabilities for pre-op patient counseling.
"""

from __future__ import annotations
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH as CoxPH_NN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
np.random.seed(42)
torch.manual_seed(42)

# --------------------------------------------------------------------------
# Load + label
# --------------------------------------------------------------------------
df = pd.read_csv(ROOT / "cleaned_data.csv")
df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)

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
    "T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1",
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
]

X = df[present].copy()
for c in numeric_like:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
str_cats = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=str_cats, drop_first=True).fillna(0.0)

times = df["time_surv"].astype(float).values
events = df[event_col].astype(bool).values
print(f"Cohort: {len(df)} patients, {int(events.sum())} events, {X_encoded.shape[1]} encoded cols")

# --------------------------------------------------------------------------
# Same preprocessing the RSF bundle uses (so the wire format stays identical)
# --------------------------------------------------------------------------
binary_cols = [c for c in X_encoded.columns if set(np.unique(X_encoded[c])) <= {0, 1}]
rare_cols = [c for c in binary_cols if X_encoded[c].sum() < 5]
print(f"Dropping {len(rare_cols)} ultra-rare binary cols")
X_reduced = X_encoded.drop(columns=rare_cols)

numeric_present = [c for c in numeric_like if c in X_reduced.columns]
scaler = StandardScaler().fit(X_reduced[numeric_present].values)
n_pca = min(3, len(numeric_present))
pca = PCA(n_components=n_pca, random_state=42).fit(
    scaler.transform(X_reduced[numeric_present].values)
)
pcs = pca.transform(scaler.transform(X_reduced[numeric_present].values))
df_pcs = pd.DataFrame(pcs, columns=[f"pca_num_{i+1}" for i in range(n_pca)],
                      index=X_reduced.index)
X_final = pd.concat([X_reduced.drop(columns=numeric_present), df_pcs], axis=1)
feature_cols = X_final.columns.tolist()
X_vals = X_final.values.astype("float32")
print(f"Final feature matrix: {X_vals.shape}")

# --------------------------------------------------------------------------
# DeepSurv (same architecture used in CV)
# --------------------------------------------------------------------------
in_features = X_vals.shape[1]
NUM_NODES = [32, 16]
DROPOUT = 0.5

net = tt.practical.MLPVanilla(
    in_features=in_features,
    num_nodes=NUM_NODES,
    out_features=1,
    batch_norm=True,
    dropout=DROPOUT,
    output_bias=False,
)
model = CoxPH_NN(net, tt.optim.Adam(lr=1e-3))

print("Training DeepSurv (80 epochs, batch 64)...")
model.fit(
    X_vals,
    (times.astype("float32"), events.astype("float32")),
    batch_size=64,
    epochs=80,
    verbose=False,
)
model.compute_baseline_hazards()
print("Done. Baseline hazards computed.")

# Sanity: training-set risk score and 24-month survival range
risk = model.predict(X_vals).reshape(-1)
surv_df = model.predict_surv_df(X_vals)
s24 = np.interp(24.0, surv_df.index.values.astype(float), surv_df.iloc[:, 0].values)
print(f"  Cohort risk score range: [{risk.min():.3f}, {risk.max():.3f}]")
print(f"  Patient 0 survival at 24mo: {s24:.3f}")

# --------------------------------------------------------------------------
# Save bundle
# --------------------------------------------------------------------------
os.makedirs(ROOT / "models", exist_ok=True)
bundle = {
    # Network rehydration:
    "state_dict": model.net.state_dict(),
    "in_features": in_features,
    "num_nodes": NUM_NODES,
    "dropout": DROPOUT,
    # Baseline hazards (pycox needs these to predict survival functions):
    "baseline_hazards": model.baseline_hazards_,
    "baseline_cumulative_hazards": model.baseline_cumulative_hazards_,
    # Preprocessing pipeline (identical to RSF bundle):
    "feature_columns": feature_cols,
    "encoded_columns": X_encoded.columns.tolist(),
    "rare_cols": rare_cols,
    "numeric_present": numeric_present,
    "scaler": scaler,
    "pca": pca,
    # Cohort risk vector for percentile (DeepSurv-specific scale):
    "cohort_risk": risk,
}

out_path = ROOT / "models" / "deepsurv_bundle.pkl"
with open(out_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"\nSaved -> {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")
