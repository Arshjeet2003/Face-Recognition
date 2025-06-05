# Face Recognition Web App

A web application built with Flask that allows users to upload an image, select a pretrained face recognition model, and get predictions in real-time. The app uses TensorFlow/Keras for model inference, OpenCV Haar Cascade for face detection, and a responsive Bootstrap frontend with AJAX to deliver smooth user experience without page reloads.

---

## Features

- Supports multiple pretrained models: Inception-ResNet, DeepID, ArcFace, FaceNet.
- Automatically detects if an uploaded image contains a face.
- Real-time predictions with AJAX to prevent page reload.
- Loading spinner to indicate processing.
- Detailed server-side logging for debugging and monitoring.
- Displays uploaded image alongside prediction result.

---

## Requirements

- Python 3.10+
- TensorFlow 2.12
- Keras 2.12
- Flask 2.2.5
- OpenCV (opencv-python-headless) 4.7.0.72
- NumPy >=1.22, <1.24

---

## Installation

1. Clone the repository:

2. Place pretrained TensorFlow/Keras models in the models/ directory from:
    - Inception_ResNet (Accuracy - 97.5%): https://drive.google.com/drive/folders/17o8BelDszOfdoon4u5nii0VGrUAvPF-k?usp=sharing
    - ArcFace (Accuracy - 84%): https://drive.google.com/drive/folders/191CEYc-OPpoPTg5F3T0yOZpaNZvJTQ1u?usp=sharing
    - FaceNet (Accuracy - 80%): https://drive.google.com/drive/folders/1R-eL_trRag-rgCUp8hvFUF9ni0Mo5-OR?usp=sharing
    - DeepID (Accuracy - 71%): https://drive.google.com/drive/folders/1DGW80HZfM6aNxgoWNZ2ZffiR9V4cDJCS?usp=sharing

3. Move to the project folder

4. Start the docker engine and run (build image): 
```bash docker build -t face_recognition .```

5. Start container at port 5000:
```bash docker run -it --rm -p 5001:5000 face_recognition```

## Usage

- Open the web app in your browser.
- Select one of the available face recognition models from the dropdown.
- Upload an image containing a face.
- Click Predict.
- View the prediction and uploaded image without page reload.

## Logging

The app outputs detailed logs to the console, including info about:

- Image uploads
- Face detection results
- Model loading
- Prediction results
- Errors and exceptions

You can customize logging handlers in app.py to log into files if needed.
