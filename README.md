# 🍎 Fruit Classifier 🥭

## Overview

The Fruit Classifier is a Streamlit web application that uses machine learning models to classify images of fruits as either fresh or rotten. This project demonstrates the practical application of deep learning in the field of agriculture and food quality assessment.

## Features

- Upload images of fruits for classification
- Choose between two different classification models:
  - Creatus Model
  - Teachable Machine Model
- Real-time prediction of fruit freshness
- Informative sidebar with model accuracy information and project details
- Help section explaining the app's functionality and components

## Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- PIL (Python Imaging Library)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/fruit-classifier.git
   cd fruit-classifier
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run ModelInvoker-NWivS2WY2mkQ2cD7gqnYsVNCVAaPMo.py
   ```

## Usage

1. Open the app in your web browser (usually at `http://localhost:8501`).
2. Select a model from the dropdown menu in the sidebar.
3. Upload an image of a fruit using the file uploader.
4. The app will process the image and display the prediction (fresh or rotten).
5. Use the "Get Help" button in the sidebar for more information about the app's functionality.

## Models

The app includes two different models for fruit classification:

1. **Creatus Model**: A custom-built model with high accuracy for low-difficulty classifications.
2. **Teachable Machine Model**: A model created using Google's Teachable Machine, offering medium accuracy.

## Project Structure

- `ModelInvoker-NWivS2WY2mkQ2cD7gqnYsVNCVAaPMo.py`: Main Streamlit application file
- `fruit_classifier_Creatus.tflite`: TensorFlow Lite model file for the Creatus model
- `fruit_classifier_teachable_machine.tflite`: TensorFlow Lite model file for the Teachable Machine model
- `requirements.txt`: List of Python dependencies

## Contributing

Contributions to improve the Fruit Classifier are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Creator

Created by Pranav Lejith (Amphibiar)(Developer)

## Acknowledgments

- TensorFlow and Keras communities
- Streamlit for their hosting framework
- Google's Teachable Machine platform
