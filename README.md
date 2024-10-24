# COVID-19 Detection using CNN
This project is about construction of a basic CNN using Keras for the detection of COVID-19 Pneumonia from chest X-ray images. This trained model is integrated using Flask API into a web app developed using JQuery, JavaScript that detects COVID-19 when providing a chest x-ray image of a patient. 

## Features

- CNN model built with Keras for COVID-19 detection
- Web interface for image upload and prediction
- Real-time prediction results
- Support for various image formats


## Tech Stack

- **Deep Learning Framework:**
  - Keras
  - TensorFlow
  - OpenCV

- **Backend:**
  - Python
  - Flask
  - NumPy
  - Pillow

- **Frontend:**
  - HTML
  - JavaScript
  - jQuery
  - CSS

## Model Architecture

The CNN model uses a parallel architecture with multiple kernel sizes:
- Parallel convolution layers with kernel sizes 3x3, 5x5, and 7x7
- Multiple Conv2D layers with MaxPooling
- Dropout layers for preventing overfitting
- Dense layers with softmax activation for final classification

## Model Performance

The model is trained on chest X-ray images and can classify between:
- COVID-19 Positive
- COVID-19 Negative

## Note

This tool is for research and educational purposes only and should not be used as a substitute for professional medical diagnosis.
