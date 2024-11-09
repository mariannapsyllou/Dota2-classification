import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load models
with open("basic_model.pkl", "rb") as f:
    basic_model = pickle.load(f)

with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Page setup
st.set_page_config(page_title="Hero Team Prediction App", layout="wide")
st.title("Hero Team Selection and Outcome Prediction")

# File uploader for hero data
uploaded_file = st.file_uploader("Upload JSON File Containing Heroes Data", type="json")

if uploaded_file:
    data = json.load(uploaded_file)
    df_heroes = pd.DataFrame(data['heroes'])

    # Use the localized names for compatibility with the model's feature names
    hero_names = df_heroes['localized_name'].tolist()
    model_feature_names = list(basic_model.feature_names_in_)

    # Filter hero_names to ensure compatibility with model features
    hero_names = [name for name in hero_names if name in model_feature_names]

    st.write("### Heroes Data Sample")
    st.write(df_heroes[['localized_name', 'id', 'name']].head())

    # Team selection
    st.write("## Select 5 Unique Heroes for Team A and Team B")
    team_a = st.multiselect("Choose 5 unique heroes for Team A", hero_names, key='team_a', max_selections=5)
    available_for_team_b = [hero for hero in hero_names if hero not in team_a]
    team_b = st.multiselect("Choose 5 unique heroes for Team B (different from Team A)", available_for_team_b, key='team_b', max_selections=5)

    if len(team_a) == 5 and len(team_b) == 5:
        st.write(f"**Team A:** {team_a}")
        st.write(f"**Team B:** {team_b}")

        # Prepare data for prediction
        features = np.zeros(len(model_feature_names))
        for hero in team_a + team_b:
            if hero in model_feature_names:
                idx = model_feature_names.index(hero)
                features[idx] = 1

        # Create DataFrame for prediction with correct column names
        features_df = pd.DataFrame([features], columns=model_feature_names)

        # Choose model and make prediction
        model_choice = st.radio("Choose a model for prediction", ["Basic Model", "Optimized Model"])
        if model_choice == "Basic Model":
            prediction = basic_model.predict(features_df)
        else:
            prediction = best_model.predict(features_df)

        # Display the prediction result
        st.subheader("Prediction Outcome")
        if prediction[0] == 1:
            st.write("Prediction: Team A is likely to win!")
        else:
            st.write("Prediction: Team B is likely to win!")
    else:
        st.warning("Please select exactly 5 unique heroes for each team.")

