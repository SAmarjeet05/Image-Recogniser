# 🖼️ Image Recogniser

🚀 Live Demo: https://image-recogniser.streamlit.app/

An interactive web application for image recognition using the **MobileNetV2** deep learning model. The model comes pre-trained on the **ImageNet** dataset, allowing it to correctly identify up to 1,000 different object categories in photographs without requiring custom dataset training or high-performance GPUs.

---

## 📂 Project Structure

```
Image Recogniser/
│
├── dataset/
│   └── imagenet_class_index.json   <- Local copy of the 1,000 ImageNet categories
│
├── Images/
│   ├── sample_dog.png              <- Sample image for testing (golden retriever)
│   ├── sample_car.png              <- Sample image for testing (sports car)
│   └── sample_pizza.png            <- Sample image for testing (pizza)
│
├── notebook/
│   └── image_recognition.ipynb     <- Step-by-step pipeline notebook
│
├── app.py                          <- Streamlit application code
├── model.py                        <- Preprocessing and prediction utilities
├── README.md                       <- Project documentation
├── requirements.txt                <- Project dependencies
└── .gitignore                      <- Git ignore rules
```

---

## 🚀 Getting Started

### 1. Clone the repository and navigate to it:
```bash
git clone <repository-url>
cd "Image Recogniser"
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Initialize Model and Cache Data
Run the helper script directly to pre-download the MobileNetV2 model weights (~14MB) and the ImageNet class index JSON so they are cached locally for offline use:
```bash
python model.py
```

### 4. Launch the Web Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
Open your browser and navigate to the local address provided (typically `http://localhost:8501`).

---

## 🛠️ Image Preprocessing Pipeline

To recognize objects accurately, raw photos are preprocessed through these steps:
1.  **Resizing:** Scales the image resolution to exactly `224x224` pixels.
2.  **Color Channels:** Ensures image is formatted in standard `RGB` space.
3.  **Dimensional Expansion:** Adds a batch dimension, shaping the array to `(1, 224, 224, 3)`.
4.  **Normalisation:** Scales pixel values from `[0, 255]` to the `[-1, 1]` intensity range expected by MobileNetV2's convolutional filters.
