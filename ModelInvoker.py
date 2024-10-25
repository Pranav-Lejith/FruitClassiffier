import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Define the help content with added color
HELP_CONTENT = """
<style>
h2 {color: #4CAF50;}
h3 {color: #2196F3;}
strong {color: #FF9800;}
red {color: #C7253E;}
</style>
<h2>Fruit Classifier App Explanation</h2>
<p>This Streamlit app classifies fruits as fresh or rotten using TensorFlow Lite models. Here's how it works:</p>
<h3>Key Components:</h3>
<ul>
    <li><strong>TensorFlow and Keras:</strong> Used for loading and running the machine learning models.</li>
    <li><strong>Streamlit:</strong> Provides the web interface for the application.</li>
    <li><strong>PIL (Python Imaging Library):</strong> Used for image processing.</li>
</ul>
<h3>Main Functions:</h3>
<ul>
    <li><strong>load_model():</strong> Loads the TensorFlow Lite model.</li>
    <li><strong>get_class_labels():</strong> Returns the class labels for the selected model.</li>
    <li><strong>get_image_size():</strong> Returns the required image size for the selected model.</li>
    <li><strong>prepare_image():</strong> Resizes and preprocesses the uploaded image for the model.</li>
</ul>
<h3>App Flow:</h3>
<ol>
    <li>User selects a model (Creatus or Teachable Machine) from the sidebar.</li>
    <li>User uploads an image of a fruit.</li>
    <li>The app preprocesses the image to match the model's requirements.</li>
    <li>The selected model makes a prediction (fresh or rotten).</li>
    <li>The result is displayed to the user.</li>
</ol>
<h3>Additional Features:</h3>
<ul>
    <li>Sidebar with model accuracy information and project details.</li>
    <li>Custom page configuration with title and icon.</li>
</ul>
<p>Created by <strong>Pranav Lejith (Amphibiar)<red> (IX-K)</red> </strong> .</p>
"""

# Initialize session state
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

# Set page config
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon='✨',
    menu_items={
        'About': "# :red[Creator]:blue[:] :violet[Pranav Lejith(:green[Amphibiar])]"
    }
)

# Function to load the TensorFlow Lite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to get class labels based on the selected model
def get_class_labels(model_selection):
    return {0: 'Fresh Fruit', 1: 'Rotten Fruit'}

# Function to get model-specific image size
def get_image_size(model_selection):
    if model_selection == "Teachable Machine Model":
        return (224, 224)  # Example for this model
    else:
        return (64, 64)  # Example for the Creatus model

# Function to prepare the image for the model
def prepare_image(image, model_selection):
    image_size = get_image_size(model_selection)  # Dynamically get the image size based on the model
    image = image.resize(image_size)
    image = img_to_array(image)

    # Ensure the image has the correct dimensions (e.g., 64x64x3 or 224x224x3)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image shape mismatch: expected {image_size + (3,)}, but got {image.shape}")
    
    # Reshape to include the batch dimension
    image = np.expand_dims(image, axis=0)  # Shape will be (1, image_size[0], image_size[1], 3)
    
    # Normalize the image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    
    return image.astype(np.float32)

# Sidebar content
with st.sidebar:
    st.title(" :violet[Choose the model]")
    
    # Model selection moved to sidebar
    model_selection = st.selectbox(
        "Model",
        ("Creatus Model", "Teachable Machine Model")
    )

    with st.expander("Model Accuracy Information"):
        st.write("""
        <style>
        .dataframe {
            border-collapse: collapse;
            width: 100%;
        }
        .dataframe th, .dataframe td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .dataframe th {
            background-color: #f2f2f2;
            text-align: left;
        }
        </style>
        """, unsafe_allow_html=True)

        st.write("""
        The following table provides information about the accuracy of the models:

        | Model                               | Accuracy       | Classes         |
        |-------------------------------------|----------------|-----------------|
        | Teachable Machine Model             | Medium         | Fresh, Rotten   |
        | Creatus Model                       | High(Low-diff) | Fresh, Rotten   |
        """)

    st.sidebar.title("🌟 :green[About the Project]")
    st.sidebar.write("""
    This project uses machine learning models to classify images of fruits as either fresh or rotten.

    Created by **:red[Pranav Lejith] (:green[Amphibiar])**.
                 
    Created for AI Project.
    """)

    st.sidebar.title("💡 :blue[Note]")
    st.sidebar.write("""
    This model is still in development and may not always be accurate. Please ensure the image is clear and well-lit for better results.
    """)

    st.sidebar.title("🛠️ :red[Functionality]")
    st.sidebar.write("""
    This AI model uses convolutional neural networks (CNNs) to analyze images of fruits. The model has been trained to classify fruits into two categories: fresh or rotten.
    """)

    # "Get Help" button moved to the bottom of the sidebar
    if st.button("Get Help"):
        st.session_state.show_help = True

# Display help content if triggered
if st.session_state.show_help:
    st.markdown(HELP_CONTENT, unsafe_allow_html=True)
    if st.button("Close Help"):
        st.session_state.show_help = False
    st.stop()  # Stop further execution to show only the help content

# Main content
st.title("🍎 :violet[Fruit Classifier] 🥭")
st.write("Upload an image to classify the fruit as fresh or rotten.")

# Model paths (Update the paths to your models)
if model_selection == "Teachable Machine Model":
    model_path = "fruit_classifier_teachable_machine.tflite"
else:
    model_path = "fruit_classifier_Creatus.tflite"

# Load the selected model
interpreter = load_model(model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the class labels based on the model
class_labels = get_class_labels(model_selection)

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prepare the image for the model
    prepared_image = prepare_image(image, model_selection)

    # Set the tensor for input
    interpreter.set_tensor(input_details[0]['index'], prepared_image)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)
    predicted_class = class_labels.get(predicted_class_index, "Unknown")

    st.write(f"🚀 The predicted class of fruit is: **{predicted_class}**")