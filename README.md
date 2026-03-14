Module Name - Computer Vision
Module ID - NB6010CEM
Student ID - COBSCCOMP4Y241P-026

Plant Health Monitor – Plant Disease Recognition System

1. Project Overview

The Plant Health Monitor is a computer vision–based application designed to detect plant diseases from leaf images. The system uses image preprocessing techniques and a Convolutional Neural Network (CNN) to classify plant leaves as healthy or diseased.

The goal of this project is to demonstrate how deep learning and image processing techniques can be applied to agricultural problems, helping farmers identify plant diseases quickly and accurately.

2. Objectives

Automatically detect plant diseases from leaf images
Apply basic image preprocessing techniques
Train a lightweight CNN model for image classification
Provide a simple web interface for uploading leaf images and viewing predictions

3. Technologies Used

Python
TensorFlow / Keras
NumPy
Matplotlib
OpenCV
Flask (Web Interface)

5. Project Structure

plant_health_monitor/
│
├── dataset/
│   ├── train/
│   ├── val/
│
├── static/
│
|── ontology/
│   └── agriculture_ontology.json
|
├── templates/
│   └── index.html
│
├── model/
│   └── mobilenet_model.pth
|   ├── resnet_model.pth
│
├── app.py
├── train_resnet.py
├── train.py
├── llm_agent.py
├── evaluate.py
├── requirements.txt
└── README.md

6. Dataset

The dataset consists of plant leaf images labeled as healthy or diseased.

Source: Kaggle Plant Disease Dataset

The images are organized into class-based folders, allowing the CNN model to learn different disease patterns effectively.

7. Running the Application

Start the Flask server:
python app.py

Open your browser and go to:
http://127.0.0.1:5000

Upload a plant leaf image to receive a prediction.

8. Model Architecture

The CNN model consists of the following layers:
Convolution Layer (feature extraction)
Max Pooling Layer (dimension reduction)
Additional Convolution + Pooling Layers
Flatten Layer
Fully Connected Dense Layers
Output Layer for classification
The model is trained using the Adam optimizer and cross-entropy loss function.
