# 🚦 Pick the Safer Road

**Interactive Road Accident Risk Prediction Game using Streamlit**

---

## Overview

This Streamlit app allows users to explore road safety by comparing two roads and predicting which is safer based on road, traffic, and environmental features. The app can be used as an interactive game, educational tool, or demo for accident risk prediction models.

It was designed for the Kaggle Playground Series / Stack Overflow Code Challenge #10.

---

## Features

- **Interactive Road Inputs**  
  - Numeric: `num_lanes`, `curvature`, `speed_limit`, `traffic_volume`, `visibility`  
  - Categorical: `road_type`, `lighting`, `weather`, `time_of_day`, `road_signs_present`, `public_road`, `holiday`, `school_season`  
  - Randomize buttons generate realistic random values for each road.

- **Prediction Models**  
  - Loads pre-trained ML models: LightGBM, CatBoost, XGBoost (if available).  
  - Supports **ensemble predictions** (average) or **single model predictions**.

- **Fallback Prediction**  
  - If models are unavailable, a weighted formula predicts accident risk based on normalized features.  
  - Corrected feature orientation and weighted contributions ensure realistic differences between roads.

- **Risk Display**  
  - Shows predicted accident risk for each road (0–1 scale).  
  - Highlights which road is safer.

- **User Interaction / Game Mode**  
  - Users guess the safer road.  
  - Correct guesses trigger visual feedback (balloons).  

- **Export & Save**  
  - Export predictions and features to CSV.  
  - Save submission metadata (Kaggle username, predictions, notes) with `joblib`.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Sheral18/PredictRoadAccident.git
cd PredictRoadAccident

Here is my Pick the Safer Road Application hosted on Streamlit
https://predictroadaccident.streamlit.app/


pip install -r requirements.txt

streamlit run app.py
