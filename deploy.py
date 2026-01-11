import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import matplotlib.pyplot as plt

img_size = 48

# Build the model architecture from scratch
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create model and try to load weights
@st.cache_resource
def load_model():
    try:
        model = create_model()
        model.load_weights("AIGeneratedModel.h5")
        return model
    except:
        st.warning("Could not load pre-trained weights. Using untrained model.")
        return create_model()


@st.cache_resource
def load_content_model():
    with st.spinner('Loading advanced image recognition model...'):
        return EfficientNetB0(weights='imagenet')

def make_gradcam_heatmap(img_array, model):
    try:
        # Convert to TensorFlow variable to ensure gradients work
        img_tensor = tf.Variable(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = model(img_tensor)
            loss = predictions[:, 0]
        
        # Get gradients
        grads = tape.gradient(loss, img_tensor)
        
        if grads is None:
            return None
            
        # Process gradients
        grads = tf.abs(grads)
        grads = tf.reduce_mean(grads, axis=-1)
        grads = grads[0]  # Remove batch dimension
        
        # Normalize
        grads_min = tf.reduce_min(grads)
        grads_max = tf.reduce_max(grads)
        if grads_max > grads_min:
            grads = (grads - grads_min) / (grads_max - grads_min)
        
        return grads.numpy()
    except:
        return None


def describe_image_content(image):
    try:
        # Ensure RGB format for content analysis
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        content_model = load_content_model()
        predictions = content_model.predict(img_array, verbose=0)
        decoded = decode_predictions(predictions, top=3)[0]
        
        # Create descriptive paragraph with better filtering
        items = [label.replace('_', ' ').lower() for _, label, prob in decoded if prob > 0.1]
        
        if len(items) >= 2:
            description = f"This image shows {items[0]}, and possibly {items[1]}"
            if len(items) > 2:
                description += f" or {items[2]}"
        elif len(items) == 1:
            description = f"This image shows {items[0]}"
        else:
            # Show top prediction even if confidence is low
            top_item = decoded[0][1].replace('_', ' ').lower()
            description = f"This image might contain {top_item}, though the model is uncertain"
            
        return description + "."
    except Exception as e:
        return "Unable to analyze image content."


model = load_model()

st.title("AI Image Classifier")       
        
img = st.file_uploader("Upload your Image")

if img and st.button("Check"):
    image = Image.open(img)
    st.image(img)
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = ImageOps.fit(image, (48,48), Image.Resampling.LANCZOS)
    img_array = img_to_array(image)
    new_arr = img_array/255
    test = np.array([new_arr])
    
    # Run prediction first to build the model
    y = model.predict(test)
    confidence = float(y[0][0])
    
    if confidence <= 0.5:
        st.write(f"*The given image is Real.* (Confidence: {(1-confidence)*100:.1f}%)")
    else:
        st.write(f"*The given image is AI Generated.* (Confidence: {confidence*100:.1f}%)")
    
    # Add simple explanation
    st.subheader("Model Explanation")
    
    # Show prediction confidence breakdown
    st.write("*Prediction Analysis:*")
    real_prob = (1 - confidence) * 100
    ai_prob = confidence * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Real Image Probability", f"{real_prob:.1f}%")
    with col2:
        st.metric("AI Generated Probability", f"{ai_prob:.1f}%")
    
    # Show image analysis
    st.write("*Image Features:*")
    st.write(f"- Image size: {image.size} (resized to 48x48 for analysis)")
    st.write(f"- Color mode: {Image.open(img).mode}")
    
    # Analyze image content
    st.write("*What's in the image:*")
    content_description = describe_image_content(Image.open(img))
    st.write(content_description)
    
    # Add GradCAM explanation
    st.subheader("Visual Explanation")
    
    st.write("*GradCAM Heatmap*")
    st.caption("Red areas show where the model focuses most when making its decision")
    heatmap = make_gradcam_heatmap(test, model)
    if heatmap is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        
        ax1.imshow(test[0])
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(test[0])
        ax2.imshow(heatmap, alpha=0.6, cmap='jet')
        ax2.set_title("Focus Areas (Red = High Attention)")
        ax2.axis('off')
        
        st.pyplot(fig)
    else:
        st.write("Visual explanation not available for this model")
    
    # Simple feature explanation
    if confidence > 0.7:
        st.write("🤖 *High AI confidence* - The model detected patterns typical of AI-generated images")
    elif confidence < 0.3:
        st.write("📷 *High Real confidence* - The model detected patterns typical of real photographs")
    else:
        st.write("🤔 *Uncertain* - The model found mixed patterns, making classification difficult")
