import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid
import altair as alt

# --- 1. Page Config ---
st.set_page_config(
    page_title="Digit Recognizer AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Professional CSS Styling (HEADER REMOVAL ADDED) ---
st.markdown("""
<style>
    /* 1. HIDE STREAMLIT DEFAULT HEADER & FOOTER */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 2. Adjust Top Padding to fill the gap */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* 3. Floating Name Badge (Top Right) - Adjusted Z-Index */
    .floating-badge {
        position: fixed;
        top: 20px;
        right: 25px;
        background-color: white;
        padding: 8px 15px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
        color: #333;
        z-index: 99999; /* Super high so nothing hides it */
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge-icon {
        background-color: #1E88E5; /* Matched with Blue Theme */
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 12px;
    }

    /* 4. Main Title */
    .main-title {
        font-size: 3.5rem !important; /* Size forcefully badha diya */
        color: #1E88E5 !important;
        text-align: center !important;
        font-weight: 800 !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1); /* Thoda shadow taaki pop kare */
    }
    .sub-text {
        text-align: center;
        color: #757575;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }

    /* 5. Card Styling */
    .card-container {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #f0f0f0;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* 6. Sidebar Accuracy Box */
    .metric-box {
        background-color: #e8f5e9;
        border-left: 6px solid #43a047;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* 7. Prediction Result Big Text */
    .prediction-box {
        background-color: #f1f8e9;
        border: 2px solid #c5e1a5;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Inject Floating Badge ---
st.markdown("""
<div class="floating-badge">
    <div class="badge-icon">S</div>
    <span>Developed by <b>Shalvi</b></span>
</div>
""", unsafe_allow_html=True)

# --- 4. Initialize Session State ---
if 'canvas_key' not in st.session_state:
    st.session_state['canvas_key'] = "canvas_1"

# --- 5. Load Models ---
@st.cache_resource
def load_models():
    models = {}
    try:
        models['Logistic Regression'] = joblib.load('model_logistic.pkl')
        models['ANN v1 (Simple)'] = tf.keras.models.load_model('model_ann.h5')
        models['ANN v2 (10 Hidden Layers)'] = tf.keras.models.load_model('model_ann2.h5')
        models['ANN v3 (Best - 100 Neurons)'] = tf.keras.models.load_model('model_ann3.h5')
        models['CNN (Convolutional)'] = tf.keras.models.load_model('model_cnn.h5')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

models_dict = load_models()

model_accuracies = {
    'Logistic Regression': "92.63%",
    'ANN v1 (Simple)': "92.50%",
    'ANN v2 (10 Hidden Layers)': "94.03%",
    'ANN v3 (Best - 100 Neurons)': "97.63%",
    'CNN (Convolutional)': "99.12%"
}

# --- 6. Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    if models_dict:
        selected_model_name = st.selectbox(
            "Select AI Model:",
            list(models_dict.keys()),
            index=4 
        )
        current_model = models_dict[selected_model_name]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Accuracy Metric Box
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-label">Model Accuracy</p>
            <p class="metric-value">{model_accuracies[selected_model_name]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("ℹ️ **Note:** The CNN model is highly recommended for best performance on handwritten digits.")
    else:
        st.error("Models failed to load.")

# --- 7. Main Layout ---

st.markdown('<h1 class="main-title">🧠 AI Digit Recognizer</h1>', unsafe_allow_html=True)

st.markdown('<p class="sub-text">Handwritten Digit Recognition using Deep Learning</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card-container"><h3>✍️ Draw Digit Here</h3></div>', unsafe_allow_html=True)
    
    c_left, c_center, c_right = st.columns([1, 6, 1])
    with c_center:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=st.session_state['canvas_key'],
            display_toolbar=False
        )
    
    if st.button('🗑️ Clear Canvas', use_container_width=True, type="primary"):
        st.session_state['canvas_key'] = str(uuid.uuid4())
        st.rerun()

with col2:
    st.markdown('<div class="card-container"><h3>🤖 AI Analysis</h3></div>', unsafe_allow_html=True)
    
    if canvas_result.image_data is not None and current_model is not None:
        img = canvas_result.image_data.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img_scaled = img_resized / 255.0
        
        if np.max(img_scaled) > 0:
            probs = None
            prediction = None
            
            # Predict Logic
            if selected_model_name == 'Logistic Regression':
                input_data = img_scaled.reshape(1, 784)
                prediction = current_model.predict(input_data)[0]
                try: probs = current_model.predict_proba(input_data)[0]
                except: probs = np.zeros(10); probs[prediction] = 1.0
                
            elif "ANN" in selected_model_name:
                if selected_model_name == 'ANN v3 (Best - 100 Neurons)':
                     input_data = img_scaled.reshape(1, 28, 28)
                else:
                     input_data = img_scaled.reshape(1, 784)
                raw_pred = current_model.predict(input_data, verbose=0)
                prediction = np.argmax(raw_pred)
                probs = raw_pred[0]
                
            elif "CNN" in selected_model_name:
                input_data = img_scaled.reshape(1, 28, 28, 1)
                raw_pred = current_model.predict(input_data, verbose=0)
                prediction = np.argmax(raw_pred)
                probs = raw_pred[0]

            # 1. Show Prediction
            st.markdown(f"""
            <div class="prediction-box">
                <p style='margin:0; font-size: 1.1rem; color: #555;'>I am {max(probs)*100:.1f}% sure this is a:</p>
                <h1 style='margin:0; font-size: 4.5rem; color: #2e7d32;'>{prediction}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. BEAUTIFUL ALTAIR CHART (Fixed numbers!)
            st.markdown("<b>Confidence Probability Chart</b>", unsafe_allow_html=True)
            
            if probs is not None:
                chart_data = pd.DataFrame({
                    "Digit": [str(i) for i in range(10)], # Strings ensure 0-9 labels show
                    "Confidence": probs,
                    "Color": ['#e0e0e0']*10 # Default Grey
                })
                # Highlight the predicted bar in Green
                chart_data.at[prediction, "Color"] = '#43a047'
                
                # Build Chart
                bar_chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Digit', sort=None, title='Digit (0-9)', axis=alt.Axis(labelFontSize=12, titleFontSize=12)),
                    y=alt.Y('Confidence', title='Confidence Score', axis=alt.Axis(format='%', labelFontSize=12)),
                    color=alt.Color('Color', scale=None, legend=None),
                    tooltip=['Digit', alt.Tooltip('Confidence', format='.1%')]
                ).properties(
                    height=250
                ).configure_axis(
                    grid=False
                )
                
                st.altair_chart(bar_chart, use_container_width=True)
                
        else:

            st.info("Waiting for drawing...")
