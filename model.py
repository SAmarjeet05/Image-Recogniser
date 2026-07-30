import os
import urllib.request
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CLASSES_PATH = os.path.join(DATASET_DIR, "imagenet_class_index.json")

def get_imagenet_classes():
    """Loads ImageNet classes from the local file or downloads it if missing."""
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            return json.load(f)
            
    # Download from official Google storage url if missing
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    os.makedirs(DATASET_DIR, exist_ok=True)
    try:
        print(f"[INFO] Downloading ImageNet class index from {url}...")
        urllib.request.urlretrieve(url, CLASSES_PATH)
        with open(CLASSES_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to download ImageNet classes: {e}")
        # Return a small static fallback of common classes if download fails completely
        fallback = {
            "263": ["n02111500", "Pembroke"],
            "285": ["n02124072", "Egyptian_cat"],
            "436": ["n03770679", "sports_car"],
            "927": ["n07873807", "pizza"]
        }
        return fallback

def load_mobilenet_model():
    """Loads and returns the pre-trained MobileNetV2 model."""
    print("[INFO] Loading pre-trained MobileNetV2 model...")
    # Load MobileNetV2 pre-trained on ImageNet weights (takes about ~14MB download on first run)
    model = MobileNetV2(weights='imagenet')
    return model

def predict_image(model, img, top_n=5):
    """Preprocesses a PIL Image and runs predictions using MobileNetV2.
    
    Returns:
        List of tuples: [(label, probability), ...]
    """
    # Ensure image is in RGB (e.g. discard alpha channel if PNG)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Resize to MobileNetV2 expected shape (224, 224)
    img_resized = img.resize((224, 224))
    
    # Convert image to numpy array
    x = np.array(img_resized, dtype=np.float32)
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    
    # Apply standard MobileNetV2 preprocessing (scales pixels between -1 and 1)
    x = preprocess_input(x)
    
    # Predict probabilities
    preds = model.predict(x)
    
    # Decode class predictions
    decoded = decode_predictions(preds, top=top_n)[0]
    
    # Process results for visual display
    results = []
    for imagenet_id, label, prob in decoded:
        formatted_label = label.replace("_", " ").title()
        results.append((formatted_label, float(prob)))
        
    return results

if __name__ == "__main__":
    # Pre-download class index when script is run directly
    print("[INFO] Initializing model helper setup...")
    classes = get_imagenet_classes()
    print(f"[SUCCESS] Loaded ImageNet classes: found {len(classes)} categories.")
    model = load_mobilenet_model()
    print("[SUCCESS] MobileNetV2 model loaded successfully!")
