from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
from sksurv.metrics import concordance_index_censored

app = FastAPI(title="ASD Survival API", version="1.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# -------------------------
# Load model bundle
# -------------------------
with open("models/rsf_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

rsf = bundle["model"]
feature_cols = bundle["feature_columns"]
encoded_columns = bundle["encoded_columns"]
rare_cols = set(bundle["rare_cols"])
numeric_present = bundle["numeric_present"]
scaler = bundle["scaler"]
pca = bundle["pca"]
X_vals = bundle["X_vals"]

# -------------------------
# Feature schema
# -------------------------
class PatientInput(BaseModel):
    Sex: int
    Age: float
    BMI: float
    prior_back_surgeries: int = 0

    dx_adjacent_segment: int = 0
    dx_spondylolisthesis: int = 0
    dx_spondylosis: int = 0
    dx_stenosis: int = 0
    dx_scoliosis: int = 0
    dx_flat_back: int = 0
    dx_sagittal_imbalance: int = 0
    dx_post_laminectomy: int = 0
    dx_deformity: int = 0

    T12_L1: int = 0
    L1_L2: int = 0
    L2_L3: int = 0
    L3_L4: int = 0
    L4_L5: int = 0
    L5_S1: int = 0

    Open: int = 0
    Perc_screws: int = 0
    Lateral_Count: int = 0
    ALIF_Count: int = 0
    ACR: int = 0

    Average_PI: float
    PI_LL_mismatch: float
    ABS_PI_LL_mismatch: float
    PI_gt_50: int = 0

    post_LL: float
    post_SVA: float
    Post_op_SS: float

    LOS: float

# -------------------------
# Utilities
# -------------------------
def build_raw_row(payload: dict):
    row = {c: 0 for c in feature_cols}
    row.update(payload)
    return pd.DataFrame([row])

def preprocess(df_raw: pd.DataFrame):
    # One-hot encode
    X = pd.get_dummies(df_raw, drop_first=True)

    # Add missing encoded columns
    for c in encoded_columns:
        if c not in X.columns:
            X[c] = 0

    # Enforce training order
    X = X[encoded_columns]

    # Drop ultra-rare
    X = X.drop(columns=[c for c in rare_cols if c in X.columns], errors="ignore")

    # PCA numeric block
    X_num = X[numeric_present].values
    X_num_scaled = scaler.transform(X_num)
    pcs = pca.transform(X_num_scaled)

    df_pcs = pd.DataFrame(
        pcs,
        columns=[f"pca_num_{i+1}" for i in range(pcs.shape[1])],
        index=X.index,
    )

    X_final = pd.concat(
        [X.drop(columns=numeric_present), df_pcs],
        axis=1,
    )

    X_final = X_final[feature_cols]
    return X_final.values.astype(float)

# -------------------------
# API endpoint
# -------------------------
@app.post("/predict/asd")
def predict_asd(inp: PatientInput):
    try:
        payload = inp.dict()

        # extract surgery text features
        surgery_feats = extract_surgery_text_features(
            payload.get("surgery_description", "")
        )
        payload.update(surgery_feats)
        X_raw = build_raw_row(payload)
        Xp = preprocess(X_raw)

        surv_fn = rsf.predict_survival_function(Xp, return_array=False)[0]
        times = surv_fn.x.tolist()
        probs = surv_fn.y.tolist()

        # Horizon probs
        def interp(t):
            return float(np.interp(t, surv_fn.x, surv_fn.y))

        horizons = {t: interp(t) for t in [12, 24, 36, 60]}

        # Median
        median = None
        for t, s in zip(surv_fn.x, surv_fn.y):
            if s <= 0.5:
                median = float(t)
                break

        # Risk percentile
        cohort_risk = rsf.predict(X_vals)
        patient_risk = rsf.predict(Xp)[0]
        percentile = float((cohort_risk < patient_risk).mean() * 100)

        return {
            "survival_curve": {
                "times": times,
                "probs": probs,
            },
            "asd_free_prob": horizons,
            "median_time_months": median,
            "risk_percentile": percentile,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_surgery_text_features(text: str):
    t = (text or "").lower()

    return {
        "revision_surgery": int(
            bool(
                ("revision" in t)
                or ("hardware removal" in t)
                or ("removal of hardware" in t)
            )
        ),

        "deformity_case_text": int(
            bool(
                ("deformity" in t)
                or ("flat back" in t)
                or ("sagittal imbalance" in t)
                or ("scoliosis" in t)
            )
        ),

        "llif_or_lateral_text": int(
            ("llif" in t) or ("lateral" in t)
        ),

        "xlif_text": int(
            "xlif" in t
        ),

        "alif_text": int(
            "alif" in t
        ),

        "anterior_posterior_approach": int(
            ("anterior" in t) and ("posterior" in t)
        ),

        "osteotomy_text": int(
            ("osteotomy" in t) or ("pso" in t)
        ),
    }
