import streamlit as st
from PIL import Image, TiffImagePlugin
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import tifffile as tiff
from io import BytesIO

TiffImagePlugin.DEBUG = True


def preprocess_image(sat_image):
    """
    Preprocesses the image to make it match the DenseNet121 input requirements
    Args:
        sat_image: PIL.Image, the uploaded file

    Returns:
        A preprocessed NumPy array of shape (1,224,224,3)

    """

    # Convert the image to RGB
    if sat_image.mode != "RGB":
        sat_image = sat_image.convert("RGB")

    # Resize the image to 224x224
    sat_image = sat_image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(sat_image)

    # Normalize to match DenseNet121
    image_array = preprocess_input(image_array)

    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# Loading the trained model
model_path = r"C:\Users\eyram\Documents\Deep Learning\flood_detection_model.keras"
model = load_model(model_path)

# Defining the title and description
st.title("Flood Detection from Satellite Images")
st.write("Upload a satellite image to predict floods.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    try:
        # Convert the uploaded file to a BytesIO stream
        file_bytes = BytesIO(uploaded_file.getvalue())

        # Use tifffile for better TIFF support
        image_array = tiff.imread(file_bytes)

        # If the image has 4 channels (e.g., RGBA), convert to RGB
        if image_array.shape[-1] == 4:
            image_array = image_array[..., :3]

        # Convert to 8-bit if the data type is not uint8
        if image_array.dtype != np.uint8:
            image_array = (image_array / np.iinfo(image_array.dtype).max * 255).astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(image_array)

        # Display the uploaded image
        st.image(image, caption="Upload Image", use_container_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict using the preprocessed image
        prediction = model.predict(processed_image)
        flood_prob = prediction[0][1]
        no_flood_prob = prediction[0][0]

        # Display the results
        st.write(f"Probability of Flood: **{flood_prob * 100:.2f}%**")
        st.write(f"Probability of No Flood: **{no_flood_prob * 100:.2f}%**")

        if flood_prob > 0.5:
            st.warning("The area is likely **flooded**.")
        else:
            st.success("The area is likely **not flooded**.")

    except Exception as e:
        st.error(f"Failed to process the image {e}")
