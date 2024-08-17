import cv2
import numpy as np
import streamlit as st
import pickle

# Load the trained model
with open(r"C:\Users\Y SAI KUMAR\Downloads\Test_data\test_data\dogs\KNneighbor.pkl", 'rb') as file:
    model = pickle.load(file)

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to (20, 20) as expected by the model
    image = cv2.resize(image, (20, 20))
    # Flatten the image to create a feature vector
    image = image.flatten()
    return image

# Streamlit app
st.title("Cat and  Dog Classification using Machine Learning")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the class (0 = Cat, 1 = Dog)
    prediction = model.predict([processed_image])[0]

    # Display the image
    st.image(image, channels="RGB", caption="Uploaded Image", use_column_width=True)

    # Display the prediction
    if prediction == 0:
        st.write("It's a **Cat**! 🐱")
    else:
        st.write("It's a **Dog**! 🐶")
