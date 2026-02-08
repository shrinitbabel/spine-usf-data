
import pickle, os, numpy as np
model_path = "models/gbsa_final_best.pkl"
assert os.path.exists(model_path), "Model file missing"
with open(model_path,"rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
cols = bundle["feature_columns"]

# create median sample if FEATURE_MATRIX exists else zeros
try:
    import pandas as pd
    FEATURE_MATRIX = globals().get("X_reduced", globals().get("X_encoded"))
    if FEATURE_MATRIX is not None:
        x0 = FEATURE_MATRIX.median(axis=0).to_numpy().reshape(1,-1)
    else:
        x0 = np.zeros((1,len(cols)))
except Exception:
    x0 = np.zeros((1,len(cols)))

# get survival output robustly
try:
    arr = model.predict_survival_function(x0, return_array=True)
    arr = np.asarray(arr)
except TypeError:
    funcs = model.predict_survival_function(x0)
    if not isinstance(funcs,(list,tuple)):
        funcs = [funcs]
    xs = np.unique(np.concatenate([np.asarray(f.x) for f in funcs]))
    arr = np.zeros((len(funcs), len(xs)))
    for i,f in enumerate(funcs):
        arr[i,:] = np.interp(xs, np.asarray(f.x), np.asarray(f.y), left=1.0, right=f.y[-1])

assert arr.shape[0]==1
vals = arr[0,:]
assert np.all(np.diff(vals) <= 1e-6)
assert np.all(vals >= -1e-6) and np.all(vals <= 1.0+1e-6)
risk = model.predict(x0)
assert np.isfinite(risk).all()
print("GBSA smoke test passed. risk:", risk)
