import streamlit as st
import streamlit.components.v1 as components

def help_button():
    components.html(
        """
        <script>
        function showHelp() {
            const helpContent = `
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
                <p>Created by Pranav Lejith (Amphibiar) for an AI Project.</p>
            `;
            const popup = window.open("", "Help", "width=600,height=600");
            popup.document.write(helpContent);
        }
        </script>
        <button onclick="showHelp()" style="
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        ">Get Help</button>
        """,
        height=100,
    )
