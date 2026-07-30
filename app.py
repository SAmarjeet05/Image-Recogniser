import streamlit as st
import os
import json
import pandas as pd
from PIL import Image

from model import (
    get_imagenet_classes,
    load_mobilenet_model,
    predict_image
)

# Absolute paths relative to app.py with case-robustness
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(folder_name, file_name=None):
    folder_low = os.path.join(BASE_DIR, folder_name.lower())
    folder_cap = os.path.join(BASE_DIR, folder_name.capitalize())
    selected_folder = folder_low if os.path.exists(folder_low) else folder_cap
    if file_name:
        return os.path.join(selected_folder, file_name)
    return selected_folder

IMAGES_DIR = resolve_path("Images")

# Page configuration
st.set_page_config(
    page_title="Image Recogniser Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
<style>
    /* Gradient Background for header */
    .header-container {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    /* Card design */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.06);
    }
    /* Prediction output */
    .pred-output {
        background-color: #d1ecf1;
        border-left: 6px solid #17a2b8;
        color: #0c5460;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🖼️ Image Recogniser Dashboard</div>
    <div class="header-subtitle">Classify photographs instantly using MobileNetV2 & Transfer Learning</div>
</div>
""", unsafe_allow_html=True)

# Helper function to load model safely and cache it
@st.cache_resource
def get_cached_model():
    try:
        return load_mobilenet_model()
    except Exception as e:
        st.error(f"Failed to load MobileNetV2 model: {str(e)}")
        return None

# Load model and class index
model = get_cached_model()
classes_dict = get_imagenet_classes()

# Sidebar Controls
st.sidebar.title("🛠️ Project Controls")
if model is not None:
    st.sidebar.success("✅ MobileNetV2 Model loaded successfully!")
    st.sidebar.markdown("""
    **Model Details:**
    *   **Architecture:** `MobileNetV2`
    *   **Weights:** `ImageNet (Pre-trained)`
    *   **Size:** `~14 MB` (88 layers)
    *   **Input Size:** `224x224 pixels`
    *   **Classes Supported:** `1,000 categories`
    """)
else:
    st.sidebar.error("⚠️ Model failed to load.")

# Tabs Setup
tab_classify, tab_explorer, tab_works = st.tabs([
    "🏠 Classify Image", 
    "📊 ImageNet Class Explorer", 
    "🧠 How it Works"
])

# ----------------- TAB 1: CLASSIFY IMAGE -----------------
with tab_classify:
    st.subheader("💡 Image Classification")
    st.write("Upload a photograph or click a preset image from the gallery to run predictions.")
    
    # Preset Gallery Section
    st.markdown("### 📷 Select a Sample Preset")
    col_g1, col_g2, col_g3 = st.columns(3)
    
    selected_preset_path = None
    
    with col_g1:
        dog_path = os.path.join(IMAGES_DIR, "sample_dog.png")
        if os.path.exists(dog_path):
            st.image(dog_path, caption="Preset: Golden Retriever", use_container_width=True)
            if st.button("🐶 Test Golden Retriever", use_container_width=True):
                selected_preset_path = dog_path
                
    with col_g2:
        car_path = os.path.join(IMAGES_DIR, "sample_car.png")
        if os.path.exists(car_path):
            st.image(car_path, caption="Preset: Sports Car", use_container_width=True)
            if st.button("🏎️ Test Sports Car", use_container_width=True):
                selected_preset_path = car_path
                
    with col_g3:
        pizza_path = os.path.join(IMAGES_DIR, "sample_pizza.png")
        if os.path.exists(pizza_path):
            st.image(pizza_path, caption="Preset: Pepperoni Pizza", use_container_width=True)
            if st.button("🍕 Test Pepperoni Pizza", use_container_width=True):
                selected_preset_path = pizza_path

    st.markdown("---")
    
    # Image Uploader
    uploaded_file = st.file_uploader(
        "Or Upload your own Image (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    # Determine which image to load
    img = None
    source_name = ""
    
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            source_name = f"Uploaded File ({uploaded_file.name})"
        except Exception as e:
            st.error(f"Error opening uploaded image: {str(e)}")
    elif selected_preset_path is not None:
        try:
            img = Image.open(selected_preset_path)
            source_name = f"Preset Image ({os.path.basename(selected_preset_path)})"
        except Exception as e:
            st.error(f"Error opening preset image: {str(e)}")

    # Classification output
    if img is not None:
        col_img, col_preds = st.columns([1, 1])
        
        with col_img:
            st.markdown(f"#### 🔍 Selected Source: `{source_name}`")
            # Display image in container
            st.image(img, use_container_width=True)
            
        with col_preds:
            st.markdown("#### ⚡ Classification Probabilities")
            
            if model is None:
                st.error("Cannot run classification because the model is not loaded.")
            else:
                with st.spinner("Processing image and running neural network..."):
                    try:
                        # Get top-5 predictions
                        predictions = predict_image(model, img, top_n=5)
                        
                        top_label, top_prob = predictions[0]
                        
                        # Glowing top classification callout
                        st.markdown(f"""
                        <div class="pred-output">
                            🎯 Predicted Object: {top_label} ({top_prob * 100:.2f}% Confidence)
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("**Top-5 Predictions:**")
                        for label, prob in predictions:
                            col_lbl, col_meter = st.columns([2, 5])
                            col_lbl.write(f"**{label}**")
                            col_meter.progress(prob)
                            col_meter.write(f"`{prob * 100:.2f}%`")
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

# ----------------- TAB 2: CLASS EXPLORER -----------------
with tab_explorer:
    st.subheader("📊 ImageNet Class Explorer")
    st.write("Browse and search the 1,000 standard object categories that the pre-trained MobileNetV2 model can recognize.")
    
    # Process classes dictionary into pandas dataframe
    rows = []
    for idx, data in classes_dict.items():
        wn_id, class_name = data
        formatted_name = class_name.replace("_", " ").title()
        rows.append({
            "Class Index": int(idx),
            "WordNet ID": wn_id,
            "Category Name": formatted_name
        })
        
    df_classes = pd.DataFrame(rows).sort_values("Class Index")
    
    # Search box
    search_q = st.text_input("🔍 Search Categories (e.g. dog, cat, car, ball):", "")
    
    if search_q:
        filtered_df = df_classes[df_classes['Category Name'].str.contains(search_q, case=False, na=False)]
        st.write(f"Found `{len(filtered_df)}` matching categories out of 1,000:")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    else:
        st.write("Displaying all 1,000 classes:")
        st.dataframe(df_classes, use_container_width=True, hide_index=True)

# ----------------- TAB 3: HOW IT WORKS -----------------
with tab_works:
    st.subheader("🧠 How Transfer Learning & MobileNetV2 Works")
    
    # Mermaid Pipeline Diagram
    st.markdown("### 🛠️ Processing Pipeline")
    st.write("Below is the end-to-end data pipeline from raw image to prediction vectors:")
    st.markdown("""
    ```mermaid
    graph LR
        A[Raw Image] --> B[Resize to 224x224]
        B --> C[Rescale Pixels -1 to 1]
        C --> D[MobileNetV2 Neural Network]
        D --> E[1000 Class Probabilities]
        E --> F[Top Prediction Labels]
    ```
    """)
    
    st.markdown("---")
    
    col_ed1, col_ed2 = st.columns(2)
    
    with col_ed1:
        st.markdown("""
        #### 🖼️ Image Preprocessing
        Before feeding an image to MobileNetV2, it must undergo strict preprocessing:
        1. **Resizing:** The model expects an input shape of `(224, 224, 3)`. We scale the uploaded image down to 224x224 pixels.
        2. **Normalisation:** Pixel intensity values range from `0` to `255`. MobileNetV2 expects input values to be scaled between `-1` and `1` (which matches how the network was trained).
        """)
        
    with col_ed2:
        st.markdown("""
        #### 🚀 Transfer Learning & ImageNet
        *   **ImageNet:** A massive academic dataset containing over 14 million images mapped to 20,000+ classes. A standard subset of 1,000 classes (called the ILSVRC challenge) is used for training general classifiers.
        *   **MobileNetV2:** Designed by Google specifically for mobile and edge devices. It utilizes depthwise separable convolutions to dramatically reduce parameter size while maintaining high prediction accuracy.
        *   **Transfer Learning:** Instead of spending thousands of dollars training a model on millions of images, we reuse Google's pre-trained neural connections, utilizing its pre-extracted features to classify images instantly in real-time.
        """)
