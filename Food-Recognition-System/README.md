# Food Recognition System using Deep Learning

This project implements a **Food Recognition System using Convolutional Neural Networks (CNN)** to identify Indian food items from images.

The system uses **MobileNetV2 with transfer learning** to classify different food categories based on visual features such as color, texture, and shape.

---

## Features

- Automatic food image classification
- Deep learning model using MobileNetV2
- Image preprocessing and augmentation
- High accuracy food recognition
- Interactive prediction interface using Gradio

---

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Gradio
- NumPy
- Matplotlib

---

## Dataset

Dataset used:
Indian Food Classification Dataset from Kaggle.

It includes images of food items such as:

- Samosa
- Biryani
- Jalebi
- Idli
- Dosa
- Chapati
- Paneer dishes
- Fried rice

---

## Model Architecture

The system uses **MobileNetV2 pretrained on ImageNet** with transfer learning.

Architecture:

Input Image (224x224)  
↓  
MobileNetV2 Feature Extraction  
↓  
Global Average Pooling  
↓  
Dense Layer  
↓  
Softmax Output (Food Categories)

---

## Installation

Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/Food-Recognition-System.git