import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Pick the Safer Road", layout="wide")

# --- Load models (if available) ---
def load_models(model_dir="models"):
    models = {}
    for name in ["lgb", "cat", "xgb"]:
        path = os.path.join(model_dir, f"{name}_model.joblib")
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except:
                st.warning(f"Failed to load {name} model.")
    return models

models = load_models()

# --- Input features ---
FEATURES = {
    "num_lanes": (1, 6, 2, 1),
    "curvature": (0.0, 1.0, 0.3, 0.01),
    "speed_limit": (20, 140, 60, 5),
    "traffic_volume": (0, 10000, 2000, 100),
    "visibility": (0, 1000, 500, 10),
    "road_type": ["highway", "urban", "rural", "residential"],
    "lighting": ["daylight", "dawn_dusk", "night"],
    "weather": ["clear", "rain", "fog", "snow"]
}

# --- Fallback risk score (simple formula) ---
def simple_risk(row):
    return (
        0.15 * (1 - row["num_lanes"]/6) +
        0.25 * row["curvature"] +
        0.2  * (row["speed_limit"]/140) +
        0.2  * (row["traffic_volume"]/10000) +
        0.1  * (1 - row["visibility"]/1000) +
        0.1  * (1 if row["lighting"] == "night" else 0.3 if row["lighting"] == "dawn_dusk" else 0) +
        0.1  * (0 if row["weather"] == "clear" else 0.3 if row["weather"] == "rain" else 0.5 if row["weather"] == "fog" else 0.7)
    )

# --- Streamlit UI ---
st.title("🚦 Pick the Safer Road")

col1, col2 = st.columns(2)

roads = {}
for i, col in enumerate([col1, col2]):
    with col:
        st.subheader(f"Road {i+1}")
        values = {}
        for feat, opts in FEATURES.items():
            if isinstance(opts, tuple):
                mn, mx, df, step = opts
                values[feat] = st.slider(feat, mn, mx, df, step=step, key=f"{feat}_{i}")
            else:
                values[feat] = st.selectbox(feat, opts, key=f"{feat}_{i}")
        roads[f"road{i+1}"] = values

# --- Convert to DataFrame ---
df = pd.DataFrame(roads).T.fillna(0)

# --- Predict (fallback if no model) ---
try:
    if models:
        # just pick the first model available
        model = list(models.values())[0]
        preds = model.predict(df.select_dtypes(include=[np.number]))
    else:
        preds = df.apply(simple_risk, axis=1)
except:
    preds = df.apply(simple_risk, axis=1)

# --- Display results ---
r1, r2 = preds.iloc[0], preds.iloc[1]

colA, colB = st.columns(2)
colA.metric("Road 1 Risk", f"{r1:.4f}")
colB.metric("Road 2 Risk", f"{r2:.4f}")

if r1 < r2:
    st.success(f"✅ Road 1 is safer ({r1:.4f} vs {r2:.4f})")
elif r2 < r1:
    st.success(f"✅ Road 2 is safer ({r2:.4f} vs {r1:.4f})")
else:
    st.info(f"Both roads have equal risk ({r1:.4f})")

st.markdown("<h4 style='text-align: center;'>Made by Sheral Waskar with ❤️ using Streamlit</h4>",
    unsafe_allow_html=True)