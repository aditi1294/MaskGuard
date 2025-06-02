import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import time

# Page configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling with dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        padding: 0rem 1rem;
    }
    
    .stApp {
        background-color: #121212;
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f5f5f5;
    }
    
    .hero-container {
        background: rgba(30, 30, 30, 0.7);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin: 2rem 0;
        text-align: center;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .hero-container:hover {
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(45deg, #a5b4fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-out 0.2s both;
    }
    
    .prediction-container {
        background: rgba(30, 30, 30, 0.7);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin: 2rem 0;
        text-align: center;
        animation: fadeIn 0.5s ease-out;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-with-mask {
        background: linear-gradient(45deg, #10B981, #059669);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        animation: fadeScale 0.5s ease-out;
    }
    
    .prediction-without-mask {
        background: linear-gradient(45deg, #EF4444, #DC2626);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        animation: fadeScale 0.5s ease-out;
    }
    
    .prediction-improper-mask {
        background: linear-gradient(45deg, #F59E0B, #D97706);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        animation: fadeScale 0.5s ease-out;
    }
    
    .confidence-bar {
        background: #2a2a2a;
        border-radius: 8px;
        overflow: hidden;
        margin: 1rem 0;
        height: 12px;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #818cf8, #a5b4fc);
        border-radius: 8px;
        transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .mode-selector {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stButton > button {
        background: rgba(30, 30, 30, 0.7) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: rgba(40, 40, 40, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    .upload-area {
        border: 2px dashed rgba(129, 140, 248, 0.5);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(30, 30, 30, 0.4);
        transition: all 0.3s ease;
        margin: 1.5rem 0;
    }
    
    .upload-area:hover {
        border-color: rgba(165, 180, 252, 0.8);
        background: rgba(40, 40, 40, 0.5);
    }
    
    .stFileUploader > div > label {
        color: #a5b4fc !important;
    }
    
    .stFileUploader > div {
        color: #e0e0e0 !important;
    }
    
    .stImage img {
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stImage img:hover {
        transform: scale(1.02);
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeScale {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Override Streamlit's default styles */
    .stSpinner > div {
        border-top-color: #818cf8 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(30, 30, 30, 0.7) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stTextInput > label {
        color: #e0e0e0 !important;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(30, 30, 30, 0.7) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stSelectbox > label {
        color: #e0e0e0 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model and class labels
@st.cache_resource
def load_mask_model():
    return load_model('saved_model/mask_detector_model.h5')

model = load_mask_model()
classes = ['With Mask', 'Without Mask', 'Improper Mask']

# Hero Section - Simplified
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Face Mask Detection</div>
    <div class="hero-subtitle">
        Detect proper mask usage with AI
    </div>
</div>
""", unsafe_allow_html=True)

# Mode Selection - Simplified
col1, col2 = st.columns(2)

with col1:
    upload_selected = st.button("📁 Upload Image", key="upload_btn")

with col2:
    camera_selected = st.button("📸 Use Camera", key="camera_btn")

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = 'upload'

if upload_selected:
    st.session_state.mode = 'upload'
elif camera_selected:
    st.session_state.mode = 'camera'

def preprocess(img):
    img = img.resize((128, 128))
    img = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img

def display_prediction(label, confidence):
    # Determine prediction class for styling
    if label == 'With Mask':
        prediction_class = 'prediction-with-mask'
        emoji = '✅'
        message = 'Mask detected properly'
    elif label == 'Without Mask':
        prediction_class = 'prediction-without-mask'
        emoji = '❌'
        message = 'No mask detected'
    else:
        prediction_class = 'prediction-improper-mask'
        emoji = '⚠️'
        message = 'Mask worn improperly'
    
    st.markdown(f"""
    <div class="prediction-container">
        <div class="{prediction_class}">
            {emoji} {label}
        </div>
        <h4>Confidence: {confidence:.1f}%</h4>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence}%"></div>
        </div>
        <p style="margin-top: 1rem; color: #b0b0b0;">
            {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Upload Image Mode
if st.session_state.mode == 'upload':
    st.markdown("""
    <div class="upload-area">
        <h3>Upload an image</h3>
        <p>Drag and drop or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'png', 'jpeg'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # Show loading spinner
        with st.spinner('Processing...'):
            time.sleep(0.5)  # Slight delay for animation effect
            
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
        
        with col2:
            img = preprocess(image)
            prediction = model.predict(img)
            label = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            display_prediction(label, confidence)

# Camera Mode
elif st.session_state.mode == 'camera':
    st.markdown("""
    <div class="upload-area">
        <h3>Camera Detection</h3>
        <p>Take a photo to analyze</p>
    </div>
    """, unsafe_allow_html=True)
    
    camera_img = st.camera_input("Take a photo", label_visibility="collapsed")
    
    if camera_img:
        # Show loading spinner
        with st.spinner('Analyzing...'):
            time.sleep(0.5)  # Slight delay for animation effect
            
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(camera_img).convert('RGB')
            st.image(image, use_container_width=True)
        
        with col2:
            img = preprocess(image)
            prediction = model.predict(img)
            label = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            display_prediction(label, confidence)

# Simple Footer
st.markdown("""
<div class="footer">
    Built with Streamlit and TensorFlow
</div>
""", unsafe_allow_html=True)