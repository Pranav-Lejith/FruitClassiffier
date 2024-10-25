


import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('fruit_classifier_tensorflow_Keras.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('fruit_classifier_tensorflow_Keras.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")

