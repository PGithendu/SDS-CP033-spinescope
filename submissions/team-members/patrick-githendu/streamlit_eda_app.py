import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="SpineScope EDA & Prediction", layout="wide")

st.title("SpineScope EDA & Prediction App")
st.markdown("Explore the dataset and predict spinal condition using a neural network (TensorFlow CPU).")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("submissions/team-members/patrick-githendu/column_3C_weka.csv")
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
st.line_chart(df[selected_feature])

# --- Boxplots by Class ---
st.header("Boxplots by Class")
selected_box = st.selectbox("Select feature for boxplot", feature_cols, key="box")
box_data = df[[selected_box, 'class']]
box_chart = alt.Chart(box_data).mark_boxplot(extent='min-max').encode(
    x=alt.X('class:N', title='Class'),
    y=alt.Y(f'{selected_box}:Q', title=selected_box)
).properties(width=400, height=300)
st.altair_chart(box_chart, use_container_width=True)

# --- Correlation Matrix ---
st.header("Correlation Matrix")
corr = df[feature_cols].corr()
st.dataframe(corr)

# --- Skewness Table ---
st.header("Feature Skewness")
try:
    import scipy.stats as stats
    skewness = df[feature_cols].apply(stats.skew)
    st.dataframe(skewness.rename("Skewness"))
except ImportError:
    st.warning("scipy is not installed. Skewness statistics are unavailable.")
except Exception:
    st.warning("Could not compute skewness due to a scipy error.")

# --- Download Data ---
st.download_button("Download CSV", df.to_csv(index=False), "spinescope_data.csv")

# --- Prediction Section ---
st.header("Predict Spinal Condition")
st.markdown("Enter biomechanical measurements to predict the spinal condition using the trained neural network model (TensorFlow CPU).")

input_data = []
for col in df.columns[:-1]:
    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_data.append(val)

if st.button("Predict Condition"):
    try:
        import tensorflow as tf
        # Force TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')
        scaler_params = np.load("submissions/team-members/patrick-githendu/scaler.npz")
        scaler_mean = scaler_params["mean"]
        scaler_scale = scaler_params["scale"]
        model = tf.keras.models.load_model("submissions/team-members/patrick-githendu/best_model.h5")
        arr = np.array(input_data).reshape(1, -1)
        arr[:, 5] = np.log1p(arr[:, 5])  # log1p transform for degree_spondylolisthesis
        arr_scaled = (arr - scaler_mean) / scaler_scale
        pred = model.predict(arr_scaled)
        pred_class = np.argmax(pred, axis=1)[0]
        class_map = {0: "Hernia", 1: "Normal", 2: "Spondylolisthesis"}
        st.success(f"Predicted Condition: **{class_map.get(pred_class, 'Unknown')}**")
    except ImportError:
        st.warning("TensorFlow or required dependencies are not installed. Prediction is unavailable.")
    except FileNotFoundError as e:
        st.warning(f"Required file not found: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("Made with Streamlit for SpineScope EDA & Prediction.")