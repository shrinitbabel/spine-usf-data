from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle


# Maps each Pydantic field to the column header the trained model actually uses.
# Front-end keeps sending JSON-friendly underscored names; we translate to the
# original headers (spaces, dashes, parens, double spaces — all preserved
# exactly as they appear in cleaned_data.csv after .str.strip()).
FIELD_TO_COL = {
    "Sex": "Sex",
    "Age": "Age",
    "BMI": "BMI",
    "prior_back_surgeries": "prior back surgeries? (y=1)",

    "dx_adjacent_segment": "dx_adjacent_segment",
    "dx_spondylolisthesis": "dx_spondylolisthesis",
    "dx_spondylosis": "dx_spondylosis",
    "dx_stenosis": "dx_stenosis",
    "dx_scoliosis": "dx_scoliosis",
    "dx_flat_back": "dx_flat_back",
    "dx_sagittal_imbalance": "dx_sagittal_imbalance",
    "dx_post_laminectomy": "dx_post_laminectomy",
    "dx_deformity": "dx_deformity",

    "T12_L1": "T12-L1",
    "L1_L2": "L1-L2",
    "L2_L3": "L2-L3",
    "L3_L4": "L3-L4",
    "L4_L5": "L4-L5",
    "L5_S1": "L5-S1",

    "Open": "Open",
    "Perc_screws": "Perc screws?",
    "Lateral_Count": "Lateral Count",
    "ALIF_Count": "ALIF Count",
    "ACR": "ACR (y=1)",

    "Average_PI": "Average PI",
    "PI_LL_mismatch": "PI-LL angle mismatch",
    "ABS_PI_LL_mismatch": "ABS PI-LL angle mismatch",
    "PI_gt_50": "(1 = PI>50)",

    "post_LL": "post LL",
    "post_SVA": "post SVA",
    "post_PI": "post PI",
    "post_PT": "post PT",
    "Post_op_SS": "Post-op SS",

    "LOS": "length of hospital stay (d)",
    "Open_Check_V2": "Open Check V2",
    "Standalone_XLIF_Check": "Standalone XLIF Check",
    "Retroperitoneal_Approach_LLIF_ALIF": "Retroperitoneal Approach (LLIF ± ALIF)",
    "Anterior_Posterior_Apporoach": "Anterior + Posterior Apporoach",
    "Osteotomies_yes_no": "Osteotomies (yes/no)",

    "infection_1_yes": "infection 1=yes",
    "DVT_1_yes": "DVT  1=yes",
    "PE_1_yes": "PE  1=yes",
    "MI_1_yes": "MI 1=yes",
    "femoral_palsy_1_yes": "femoral palsy (knee extension weakness) 1=yes",
    "hip_flexion_weakness_1_yes": "hip flexion weakness (iliopsoas weakness)  1=yes",
    "acute_thigh_paresthesia": "acute thigh paresthesia (immediate post op)",
    "psoas_hematoma": "psoas hematoma",
}


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
    post_PI: float
    post_PT: float
    Post_op_SS: float

    LOS: float
    Open_Check_V2: int = 0
    Standalone_XLIF_Check: int = 0
    Retroperitoneal_Approach_LLIF_ALIF: int = 0
    Anterior_Posterior_Apporoach: int = 0
    Osteotomies_yes_no: int = 0

    infection_1_yes: int = 0
    DVT_1_yes: int = 0
    PE_1_yes: int = 0
    MI_1_yes: int = 0
    femoral_palsy_1_yes: int = 0
    hip_flexion_weakness_1_yes: int = 0
    acute_thigh_paresthesia: int = 0
    psoas_hematoma: int = 0


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("models/rsf_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    state["rsf"] = bundle["model"]
    state["feature_cols"] = bundle["feature_columns"]
    state["encoded_columns"] = bundle["encoded_columns"]
    state["rare_cols"] = set(bundle["rare_cols"])
    state["numeric_present"] = bundle["numeric_present"]
    state["scaler"] = bundle["scaler"]
    state["pca"] = bundle["pca"]
    # Score the cohort once so /predict/asd doesn't run the whole forest over
    # 546 rows on every request.
    state["cohort_risk"] = state["rsf"].predict(bundle["X_vals"])
    state["pca_loadings"] = {
        f"pca_num_{i+1}": dict(zip(state["numeric_present"], state["pca"].components_[i]))
        for i in range(state["pca"].components_.shape[0])
    }
    # Fields the wire-format exposes that don't map to any model column — log
    # them once at startup so a typo doesn't silently degrade predictions.
    encoded = set(state["encoded_columns"])
    unmapped = [f for f, c in FIELD_TO_COL.items() if c not in encoded]
    if unmapped:
        print("WARN: PatientInput fields not present in encoded_columns:", unmapped)
    yield


app = FastAPI(title="ASD Survival API", version="1.2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": "rsf" in state}


@app.get("/model-info")
def model_info():
    return {"pca_loadings": state["pca_loadings"]}


def build_raw_row(payload: dict) -> pd.DataFrame:
    # Build the row in the *encoded* column space (pre-PCA, pre-rare-drop).
    # The original app keyed off feature_cols, which had the numeric features
    # already replaced by pca_num_*, so the PCA scaler ran on zeros every call.
    row = {c: 0 for c in state["encoded_columns"]}
    for field, val in payload.items():
        col = FIELD_TO_COL.get(field)
        if col is not None and col in row:
            row[col] = val
    return pd.DataFrame([row])


def preprocess(df_raw: pd.DataFrame):
    X = df_raw[state["encoded_columns"]]
    X = X.drop(columns=[c for c in state["rare_cols"] if c in X.columns], errors="ignore")

    numeric_present = state["numeric_present"]
    X_num = X[numeric_present].astype(float).values
    X_num_scaled = state["scaler"].transform(X_num)
    pcs = state["pca"].transform(X_num_scaled)

    df_pcs = pd.DataFrame(
        pcs,
        columns=[f"pca_num_{i+1}" for i in range(pcs.shape[1])],
        index=X.index,
    )
    X_final = pd.concat([X.drop(columns=numeric_present), df_pcs], axis=1)
    X_final = X_final[state["feature_cols"]]
    return X_final.values.astype(float), df_pcs.iloc[0].to_dict()


@app.post("/predict/asd")
def predict_asd(inp: PatientInput):
    try:
        payload = inp.model_dump()
        X_raw = build_raw_row(payload)
        Xp, pca_scores = preprocess(X_raw)

        rsf = state["rsf"]
        surv_fn = rsf.predict_survival_function(Xp, return_array=False)[0]
        times = surv_fn.x.tolist()
        probs = surv_fn.y.tolist()

        def interp(t: float) -> float:
            return float(np.interp(t, surv_fn.x, surv_fn.y))

        horizons = {t: interp(t) for t in [12, 24, 36, 60]}

        median = None
        for t, s in zip(surv_fn.x, surv_fn.y):
            if s <= 0.5:
                median = float(t)
                break

        patient_risk = float(rsf.predict(Xp)[0])
        cohort_risk = state["cohort_risk"]
        percentile = float((cohort_risk < patient_risk).mean() * 100)

        return {
            "survival_curve": {"times": times, "probs": probs},
            "asd_free_prob": horizons,
            "median_time_months": median,
            "risk_percentile": percentile,
            "patient_risk": patient_risk,
            "pca_scores": pca_scores,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
