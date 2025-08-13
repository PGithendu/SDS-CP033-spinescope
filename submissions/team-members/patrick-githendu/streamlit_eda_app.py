# Streamlit EDA App for SpineScope Dataset

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SpineScope EDA", layout="wide")

st.title("SpineScope EDA Dashboard")
st.markdown("Explore the biomechanical dataset and key insights interactively.")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("column_3C_weka.csv")
    return df

df = load_data()

# --- Data Overview ---
st.header("Data Overview")
st.write(df.head())

st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))

# --- Class Distribution ---
st.subheader("Class Distribution")
st.bar_chart(df['class'].value_counts())

# --- Feature Distributions ---
st.header("Feature Distributions")
feature_cols = df.columns[:-1]
selected_feature = st.selectbox("Select a feature", feature_cols)
fig, ax = plt.subplots()
sns.histplot(df[selected_feature], kde=True, ax=ax)
st.pyplot(fig)

# --- Boxplots by Class ---
st.header("Boxplots by Class")
selected_box = st.selectbox("Select feature for boxplot", feature_cols, key="box")
fig2, ax2 = plt.subplots()
sns.boxplot(x='class', y=selected_box, data=df, ax=ax2)
st.pyplot(fig2)

# --- Correlation Matrix ---
st.header("Correlation Matrix")
corr = df[feature_cols].corr()
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# --- Pairplot (first 4 features) ---
st.header("Pairplot (First 4 Features)")
if st.button("Show Pairplot"):
    import warnings
    warnings.filterwarnings("ignore")
    fig4 = sns.pairplot(df.iloc[:, :4].assign(Label=df['class']), hue='Label')
    st.pyplot(fig4)

# --- Skewness Table ---
st.header("Feature Skewness")
import scipy.stats as stats
skewness = df[feature_cols].apply(stats.skew)
st.dataframe(skewness.rename("Skewness"))

# --- Download Data ---
st.download_button("Download CSV", df.to_csv(index=False), "spinescope_data.csv")

# --- Prediction Section ---
st.header("Predict Spinal Condition")

st.markdown("Enter biomechanical measurements to predict the spinal condition using the trained neural network model.")

# Input fields for all features
input_data = []
for col in df.columns[:-1]:
    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_data.append(val)

if st.button("Predict Condition"):
    # Load scaler and model (assumes you have saved them as 'scaler.pkl' and 'best_model.h5')
    import joblib
    import tensorflow as tf
    scaler = joblib.load("scaler.pkl")
    model = tf.keras.models.load_model("best_model.h5")
    # Preprocess input
    arr = np.array(input_data).reshape(1, -1)
    arr[:, 5] = np.log1p(arr[:, 5])  # log1p transform for degree_spondylolisthesis
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)
    pred_class = np.argmax(pred, axis=1)[0]
    class_map = {0: "Hernia", 1: "Normal", 2: "Spondylolisthesis"}
    st.success(f"Predicted Condition: **{class_map.get(pred_class, 'Unknown')}**")

st.markdown("---")
st.markdown("Made with Streamlit for SpineScope EDA.")
