# Hero Team Prediction App

This repository contains a web application for predicting the likely outcome of a match in DOTA 2 based on hero team compositions. It utilizes machine learning models to evaluate the strengths of two selected teams of heroes and outputs a prediction indicating which team has a higher probability of winning.

## Features

- Upload JSON data containing hero information.
- Select two teams of five unique heroes each from the provided dataset.
- Choose between a basic model and an optimized model for outcome prediction.
- Display the winning prediction result based on the selected team compositions.

## Requirements

To run this application, you'll need:

- Python 3.7+
- The following Python libraries:
  - `streamlit`
  - `numpy`
  - `pandas`
  - `pickle`

## Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/hero-team-prediction.git
cd hero-team-prediction
streamlit run app.py
