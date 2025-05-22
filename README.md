# Cataract Classification Using CNN-SVM
This project implements a deep learning-based approach for cataract classification using fundus images. The methodology is inspired by the research paper "Classification of Cataract Fundus Image Based on Deep Learning" by Yanyan Dong, Qinyan Zhang, Zhiqiang Qiao, and Ji-Jiang Yang.

## Project Overview
The Flask framework is used for the web application, with the implementation in info.py. This file handles the rendering of index.html and result.html.
The preprocessing functions for image enhancement and transformation are also implemented in info.py.
Templates are used to design the app interface for user interaction.
## Model Training
The main file is responsible for training the model based on the methodology described in the research paper.
In the preprocessing step, each image undergoes the following transformations:
Maximum entropy thresholding
Canny edge detection
Grayscale conversion
Data augmentation: Each image in the dataset is transformed into four additional images and saved for future use. This helps in reducing computational overhead during training.
## Deep Learning Model
The model utilizes a Convolutional Neural Network (CNN) for feature extraction.
The original research employed a 5-layer CNN, whereas this implementation uses a 13-layer CNN to enhance feature extraction capability.
Extracted features are then classified using Support Vector Machine (SVM) for improved accuracy.
This approach combines the power of deep feature extraction with SVM's robust classification capabilities to achieve high-performance cataract detection.
