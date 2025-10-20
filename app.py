import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Pick the Safer Road", layout="wide")

# --- Configuration: feature names and possible values ---
FEATURES = {
    "num_lanes": (1, 6, 2, 1),
    "curvature": (0.0, 1.0, 0.3, 0.01),
    "speed_limit": (20, 140, 60, 5),
    "traffic_volume": (0, 10000, 2000, 100),
    "visibility": (0, 1000, 500, 10),
    "road_type": ["highway", "urban", "rural", "residential"],
    "lighting": ["daylight", "dawn_dusk", "night"],
    "weather": ["clear", "rain", "fog", "snow"],
    "time_of_day": ["morning", "afternoon", "evening", "night"],
    "road_signs_present": ["yes", "no"],
    "public_road": ["yes", "no"],
    "holiday": ["no", "yes"],
    "school_season": ["no", "yes"]
}

CATEGORICAL_FEATURES = [
    "road_type", "lighting", "weather", "time_of_day",
    "road_signs_present", "public_road", "holiday", "school_season"
]

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# --- Initialize session state for roads ---
if "road1" not in st.session_state:
    st.session_state.road1 = {}
if "road2" not in st.session_state:
    st.session_state.road2 = {}

# --- Helper to build a dataframe row ---
def build_row(values):
    return pd.DataFrame({k: [v] for k, v in values.items()})

# --- Load models ---
@st.cache_resource
def load_models(model_dir="models"):
    models = {}
    candidates = {
        "lgb": os.path.join(model_dir, "lgb_model.joblib"),
        "cat": os.path.join(model_dir, "cat_model.joblib"),
        "xgb": os.path.join(model_dir, "xgb_model.joblib"),
    }
    for key, path in candidates.items():
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except:
                pass
    return models

# --- Improved fallback prediction ---
def predict_with_models(models, X):
    preds = {}
    if models:  # Use real models if available
        for k, m in models.items():
            try:
                preds[k] = m.predict(X)
            except:
                pass
        return preds

    df = X.copy()
    
    # Features normalized to [0,1], higher => higher risk
    df['lanes_risk'] = 1 - df['num_lanes']/6
    df['curvature_risk'] = df['curvature']
    df['speed_risk'] = df['speed_limit']/140
    df['traffic_risk'] = df['traffic_volume']/10000
    df['visibility_risk'] = 1 - df['visibility']/1000
    df['lighting_risk'] = df['lighting'].map(lambda x: 1.0 if x=='night' else 0.3 if x=='dawn_dusk' else 0.0)
    df['signs_risk'] = df['road_signs_present'].map(lambda x: 0.0 if x=='yes' else 0.5)
    df['weather_risk'] = df['weather'].map({'clear':0.0,'rain':0.3,'fog':0.5,'snow':0.7})

    # Weighted combination for accident risk
    df['fallback'] = (
        0.15*df['lanes_risk'] + 
        0.25*df['curvature_risk'] + 
        0.2*df['speed_risk'] +
        0.1*df['traffic_risk'] + 
        0.1*df['lighting_risk'] + 
        0.1*df['signs_risk'] +
        0.1*df['weather_risk'] +
        0.05*df['visibility_risk']
    ).clip(0,1)

    preds['fallback'] = df['fallback'].values
    return preds

# --- App Title ---
st.title("🚦 Pick the Safer Road — Road Safety Game")
st.markdown("Select road features for two roads and see which is predicted safer. Enter your Kaggle username to get the joint badge!")

# --- Sidebar: Model selection ---
models = load_models()
ensemble_method = st.sidebar.selectbox("Prediction mode", ["Ensemble (average)", "Single model (pick)"], index=0)
if ensemble_method == "Single model (pick)":
    model_choice = st.sidebar.selectbox("Pick model", ["lgb","cat","xgb"], index=0)
else:
    model_choice = None

st.sidebar.write("Loaded models:")
for k in ["lgb", "cat", "xgb"]:
    st.sidebar.write(f"- {k}: {'✅' if k in models else '❌'}")

# --- Columns for Road 1 and Road 2 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Road 1")
    for feat, val in FEATURES.items():
        if feat in CATEGORICAL_FEATURES:
            st.session_state.road1[feat] = st.selectbox(
                f"{feat}", val, key=f"r1_{feat}",
                index=val.index(st.session_state.road1.get(feat, val[0]))
            )
        else:
            mn, mx, df_val, step = val
            st.session_state.road1[feat] = st.slider(
                f"{feat}", mn, mx, st.session_state.road1.get(feat, df_val), step=step, key=f"r1_{feat}"
            )
    if st.button("Randomize Road 1"):
        for feat, val in FEATURES.items():
            if feat in CATEGORICAL_FEATURES:
                st.session_state.road1[feat] = np.random.choice(val)
            else:
                mn, mx, df_val, step = val
                choices = np.arange(mn, mx + 1e-9, step)
                st.session_state.road1[feat] = int(np.random.choice(choices)) if isinstance(df_val, int) else float(np.random.choice(choices))

with col2:
    st.subheader("Road 2")
    for feat, val in FEATURES.items():
        if feat in CATEGORICAL_FEATURES:
            st.session_state.road2[feat] = st.selectbox(
                f"{feat}", val, key=f"r2_{feat}",
                index=val.index(st.session_state.road2.get(feat, val[0]))
            )
        else:
            mn, mx, df_val, step = val
            st.session_state.road2[feat] = st.slider(
                f"{feat}", mn, mx, st.session_state.road2.get(feat, df_val), step=step, key=f"r2_{feat}"
            )
    if st.button("Randomize Road 2"):
        for feat, val in FEATURES.items():
            if feat in CATEGORICAL_FEATURES:
                st.session_state.road2[feat] = np.random.choice(val)
            else:
                mn, mx, df_val, step = val
                choices = np.arange(mn, mx + 1e-9, step)
                st.session_state.road2[feat] = int(np.random.choice(choices)) if isinstance(df_val, int) else float(np.random.choice(choices))

# --- Build dataframe ---
X = pd.concat([build_row(st.session_state.road1).assign(_id="road1"),
               build_row(st.session_state.road2).assign(_id="road2")], ignore_index=True)
X.index = ["road1", "road2"]

# --- Predictions ---
pred_dict = predict_with_models(models, X)
preds_df = pd.DataFrame(pred_dict, index=X.index)

if ensemble_method == "Ensemble (average)":
    preds_df["ensemble"] = preds_df.mean(axis=1)
elif ensemble_method == "Single model (pick)":
    if model_choice in preds_df.columns:
        preds_df["selected"] = preds_df[model_choice]
    else:
        preds_df["selected"] = preds_df.iloc[:,0]

final_col = "ensemble" if "ensemble" in preds_df.columns else ("selected" if "selected" in preds_df.columns else preds_df.columns[0])

# --- Display predictions ---
col_a, col_b = st.columns(2)
with col_a:
    st.metric("Road 1 predicted accident_risk", f"{float(preds_df.loc['road1', final_col]):.4f}")
with col_b:
    st.metric("Road 2 predicted accident_risk", f"{float(preds_df.loc['road2', final_col]):.4f}")

# Safer road info
r1_score = float(preds_df.loc['road1', final_col])
r2_score = float(preds_df.loc['road2', final_col])
if r1_score < r2_score:
    st.success(f"Road 1 is predicted to be safer ({r1_score:.4f} vs {r2_score:.4f})")
elif r2_score < r1_score:
    st.success(f"Road 2 is predicted to be safer ({r2_score:.4f} vs {r1_score:.4f})")
else:
    st.info(f"Both roads have equal predicted risk ({r1_score:.4f})")

# --- User guess ---
guess = st.radio("Which road do you think is safer?", ("Road 1", "Road 2", "Tie"))
if st.button("Submit guess"):
    correct = "Road 1" if r1_score < r2_score else ("Road 2" if r2_score < r1_score else "Tie")
    if guess == correct:
        st.balloons()
        st.success(f"Correct! ({correct})")
    else:
        st.error(f"Incorrect. Correct: {correct}")

