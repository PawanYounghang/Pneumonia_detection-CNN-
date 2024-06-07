import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model (for example, a MobileNetV2 model fine-tuned on a cat vs. dog dataset)
model = tf.keras.models.load_model('model_92_.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize the image to the input size required by the model
    image = np.array(image)  # Convert the image to a numpy array

    # Check the number of dimensions
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,)*3, axis=-1)  # Convert grayscale to RGB
    elif image.shape[2] == 4:  # RGBA image
        image = image[..., :3]  # Convert RGBA to RGB

    image = image / 255.0  # Normalize the image to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    probability = predictions[0][0]  # Get the probability of the positive class
    threshold = 0.92
    class_names = ['Normal', 'Pneumonia']  # Replace with your actual class names
    predicted_class = class_names[1] if probability >= threshold else class_names[0]
    confidence = probability if predicted_class == class_names[1] else 1 - probability
    return predicted_class, confidence

# Streamlit app
st.title("Pneumonia Detection")
st.write("Upload an image to see if it shows signs of pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class, confidence = predict(image)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")

