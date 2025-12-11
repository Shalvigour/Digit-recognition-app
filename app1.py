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

# --- 2. Professional CSS Styling ---
st.markdown("""
<style>
    /* 1. HIDE DEFAULT HEADER & FOOTER */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 2. PAGE SPACING */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* 3. STYLISH BADGE (Top Right) */
    .floating-badge {
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(8px);
        padding: 8px 15px;
        border-radius: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e3f2fd;
        z-index: 99999;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .badge-icon {
        background: #1E88E5;
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

    /* 4. TITLE STYLING */
    .main-title {
        font-size: 3.5rem !important;
        color: #1E88E5 !important;
        text-align: center !important;
        font-weight: 800 !important;
        margin-top: 10px !important;
        margin-bottom: -15px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 5. METRIC & PREDICTION BOXES */
    .prediction-box {
        background-color: #f1f8e9;
        border: 2px solid #a5d6a7;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .metric-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .card-container h3 {
        text-align: center;
        color: #333;
        padding: 10px;
        background: #f5f5f5;
        border-radius: 8px;
        margin-bottom: 15px;
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
            <p class="metric-value">{model_accuracies.get(selected_model_name, "N/A")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("ℹ️ **Note:** The CNN model is highly recommended for best performance on handwritten digits.")
    else:
        st.error("Models failed to load.")

# --- 7. Main Layout ---

# Title Section
st.markdown("""
    <div style="text-align: center; margin-top: 0px;">
        <h1 style="
            font-size: 3.5rem; 
            color: #1E88E5; 
            margin-bottom: -10px; 
            margin-top: -30px; 
            font-weight: 800;">
            🔢 AI Digit Recognizer
        </h1>
        <p style="
            font-size: 1.2rem; 
            color: #000000; 
            margin-top:-20px;
            margin-bottom:40px;
            font-weight: 350;">
            Handwritten Digit Recognition using Deep Learning
        </p>
    </div>
""", unsafe_allow_html=True)

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
        
        # 1. Contours dhoondo
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Agar koi drawing mili (Contour exist karta hai)
        if contours:
            # FIX: Combine all contours to handle disjoint numbers (like 5 or i)
            all_points = np.vstack([c for c in contours])
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Sirf digit ko crop karo
            digit_roi = img[y:y+h, x:x+w]
            
            # --- 2. Resize with Aspect Ratio (Digit ka shape mat bigado) ---
            if h > w:
                factor = 20.0 / h
                h_new = 20
                w_new = int(w * factor)
            else:
                factor = 20.0 / w
                w_new = 20
                h_new = int(h * factor)
            
            # Resize karo
            # Safety check for zero width/height
            if w_new > 0 and h_new > 0:
                digit_roi_resized = cv2.resize(digit_roi, (w_new, h_new), interpolation=cv2.INTER_AREA)
                
                # --- 3. Padding (Center mein lagao) ---
                final_img = np.zeros((28, 28), dtype=np.uint8)
                pad_y = (28 - h_new) // 2
                pad_x = (28 - w_new) // 2
                final_img[pad_y:pad_y+h_new, pad_x:pad_x+w_new] = digit_roi_resized
                
                # Normalize
                img_scaled = final_img / 255.0

                # --- PREDICTION LOGIC STARTS HERE (Inside 'if contours') ---
                probs = None
                prediction = None

                if selected_model_name == 'Logistic Regression':
                    input_data = img_scaled.reshape(1, 784)
                    prediction = current_model.predict(input_data)[0]
                    try: probs = current_model.predict_proba(input_data)[0]
                    except: 
                        probs = np.zeros(10)
                        probs[prediction] = 1.0
                        
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
                
                # 2. Altair Chart
                st.markdown("<b>Confidence Probability Chart</b>", unsafe_allow_html=True)
                
                if probs is not None:
                    chart_data = pd.DataFrame({
                        "Digit": [str(i) for i in range(10)],
                        "Confidence": probs,
                        "Color": ['#e0e0e0']*10 
                    })
                    chart_data.at[prediction, "Color"] = '#43a047'
                    
                    bar_chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X('Digit', sort=None, title='Digit (0-9)'),
                        y=alt.Y('Confidence', title='Confidence Score', axis=alt.Axis(format='%')),
                        color=alt.Color('Color', scale=None, legend=None),
                        tooltip=['Digit', alt.Tooltip('Confidence', format='.1%')]
                    ).properties(height=250)
                    
                    st.altair_chart(bar_chart, use_container_width=True)

        else:
            # Agar canvas khali hai to ye dikhao
            st.info("Waiting for drawing...")
            st.markdown("""
                <div style="text-align: center; color: #aaa; margin-top: 50px;">
                    <p>Draw a digit (0-9) in the box to see the AI prediction.</p>
                </div>
            """, unsafe_allow_html=True)
