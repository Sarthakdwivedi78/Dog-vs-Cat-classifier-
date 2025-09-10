import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import base64
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="Pet-AI Classifier 🐾",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling & Background (Self-Contained) ---
def add_bg_and_styling():
    # Base64 encoded string of a background image - no external file needed
    encoded_string = """
    iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAPMSURBVHhe7d3NbhwhEIXh7zCgZ5h3qKeYN6hnmDcoJ0hOyKk4CRAgK9ldvXbT3TBpPZ3+UqJVVb+vqqr+50yH09PT/wBw5tZ3gLMDwhCIITAIYQjEAAhDIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAxCIAbAIARiAAx-
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .stApp > header {{
            background-color: transparent;
        }}
        
        [data-testid="stSidebar"], [data-testid="stMetric"], .main-container {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        .st-emotion-cache-16txtl3 {{
            padding: 2rem 2rem;
        }}

        h1, h2, h3, p, .stMarkdown {{
            font-family: 'Poppins', sans-serif;
            color: #FFFFFF;
        }}

        [data-testid="stTab"] {{
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.08);
            margin: 0 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_and_styling()

# --- Constants & Fun Facts ---
MODEL_PATH = 'dog_cat_classifier.h5'
CLASS_NAMES = ['Cat', 'Dog']
CAT_FACTS = [
    "A group of cats is called a clowder.",
    "Cats can make over 100 different sounds, whereas dogs can only make about 10.",
    "A cat's brain is biologically more similar to a human brain than it is to a dog's.",
    "The oldest known pet cat existed 9,500 years ago.",
    "Isaac Newton is credited with inventing the cat door."
]
DOG_FACTS = [
    "A dog's sense of smell is at least 40x better than ours.",
    "Some dogs are amazing swimmers. Newfoundlands have webbed feet and a water-resistant coat.",
    "The Beatles' song 'A Day in the Life' has a frequency only dogs can hear.",
    "A dog’s nose print is unique, much like a person’s fingerprint.",
    "Three dogs survived the sinking of the Titanic."
]

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}", icon="🚨")
        return None

model = load_model()

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Core Prediction Function ---
def predict(image_source):
    try:
        image = Image.open(image_source).convert('RGB')
        img_array = np.array(image)
        img_resize = cv2.resize(img_array, (224, 224))
        img_scaled = img_resize / 255.0
        img_reshaped = np.reshape(img_scaled, [1, 224, 224, 3])
        
        prediction = model.predict(img_reshaped)[0]
        confidence = np.max(prediction)
        label_index = np.argmax(prediction)
        predicted_label = CLASS_NAMES[label_index]
        
        # Add to history (limit history to 10 items to prevent bloat)
        st.session_state.history.insert(0, (image, predicted_label, confidence))
        st.session_state.history = st.session_state.history[:10]
        
        return image, predicted_label, confidence
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}", icon="🚨")
        return None, None, None

# --- Sidebar ---
with st.sidebar:
    st.title("Pet-AI Classifier")
    st.markdown("---")
    st.subheader("About")
    st.info(
        "This app uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. "
        "Upload an image, a batch of images, or use your camera to get a real-time prediction.",
        icon="🧠"
    )
    st.markdown("---")
    st.subheader("Prediction History")
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
    
    for i, (img, label, conf) in enumerate(st.session_state.history):
        st.image(img, width=80)
        st.text(f"{i+1}. {label} ({conf:.1%})")
    st.markdown("---")


# --- Main Application ---
st.title("🐾 Is it a Cat or a Dog?")
st.write("Upload an image, a whole batch, or use your camera to find out!")

if model is None:
    st.warning("Model not loaded. Please check the error message in the console.")
    st.stop()

tab1, tab2 = st.tabs(["🖼️ Single Image", "📂 Batch Upload"])

# --- Single Image Tab ---
with tab1:
    sub_tab1, sub_tab2 = st.tabs(["Upload", "Camera"])
    
    # Use a unique variable for the file uploader
    with sub_tab1:
        uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    # Use a different, unique variable for the camera
    with sub_tab2:
        camera_picture = st.camera_input("Take a picture...")

    # Check uploaded_file separately
    if uploaded_file is not None:
        with st.spinner("Classifying..."):
            image, label, confidence = predict(uploaded_file)
            
            if image:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Your Image", use_column_width=True)
                with col2:
                    st.metric("Prediction", label, f"{confidence:.2%}")
                    fact = random.choice(DOG_FACTS) if label == 'Dog' else random.choice(CAT_FACTS)
                    st.info(f"**Fun Fact:** {fact}", icon="💡")

    # Check camera_picture separately
    if camera_picture is not None:
        with st.spinner("Classifying..."):
            image, label, confidence = predict(camera_picture)
            
            if image:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Your Image", use_column_width=True)
                with col2:
                    st.metric("Prediction", label, f"{confidence:.2%}")
                    fact = random.choice(DOG_FACTS) if label == 'Dog' else random.choice(CAT_FACTS)
                    st.info(f"**Fun Fact:** {fact}", icon="💡")

# --- Batch Upload Tab ---
with tab2:
    uploaded_files = st.file_uploader(
        "Upload multiple image files", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="batch_uploader" # Added a key for stability
    )
    if uploaded_files:
        st.write(f"### Processing {len(uploaded_files)} images...")
        
        cols = st.columns(4)
        
        for i, file in enumerate(uploaded_files):
            with cols[i % 4]:
                # Use use_container_width as per the warning
                st.image(file, use_column_width=True)
                with st.spinner("..."):
                    _, label, confidence = predict(file)
                    if label:
                        st.success(f"{label} ({confidence:.1%})")
