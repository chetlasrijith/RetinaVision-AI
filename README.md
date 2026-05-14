# Eye Disease Detection using Deep Learning

## Overview

Eye Disease Detection is a Deep Learning-based computer vision project developed to identify multiple eye diseases from retinal fundus images. The system uses Convolutional Neural Networks (CNNs) and Vision Transformer models to classify retinal scans into different disease categories with high accuracy.

The project also includes a Streamlit-based web application that allows users to upload retinal fundus images and receive real-time disease predictions.

---

## Features

* Detection of multiple eye diseases from retinal images
* Deep Learning and Vision Transformer-based models
* Streamlit web application for real-time predictions
* Large-scale dataset handling (50,000 retinal images)
* GPU-supported training pipeline
* End-to-end workflow from preprocessing to deployment

---

## Diseases Detected

The model is trained to classify the following categories:

* Age-Related Macular Degeneration (AMD)
* Cataract
* Diabetic Retinopathy (DR)
* Glaucoma
* Pathological Myopia
* Normal Fundus

---

## Dataset

The project uses the Retinal Fundus Image Dataset containing approximately 50,000 retinal images across multiple disease classes.

### Dataset Highlights

* Multi-class classification dataset
* High-resolution retinal fundus images
* Suitable for computer vision and medical imaging tasks

Dataset Source:

* Kaggle Retinal Fundus Image Dataset

---

## Tech Stack

### Languages

* Python

### Libraries & Frameworks

* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit

### Concepts Used

* Deep Learning
* Computer Vision
* Transfer Learning
* Vision Transformers
* Image Classification
* Model Deployment

---

## Project Structure

```bash
Eye-Disease-Detection/
│
├── Demo App/
│   ├── eye_disease_detection.py
│   ├── requirements.txt
│   └── model files
│
├── Final Code/
│   ├── AMD/
│   ├── Cataract/
│   ├── DR/
│   ├── Glaucoma/
│   ├── Pathological Myopia/
│   └── Vision Transformer Combine/
│
├── README.md
└── requirements.txt
```

---

## Model Training Workflow

### Steps Involved

1. Dataset Collection
2. Image Preprocessing
3. Data Augmentation
4. Model Building
5. Model Training
6. Model Evaluation
7. Model Saving
8. Deployment using Streamlit

---

## Image Preprocessing

The following preprocessing techniques were used:

* Image resizing
* Normalization
* Data augmentation
* Noise reduction
* Train-test split

---

## Model Architecture

The project uses Deep Learning models including:

* Convolutional Neural Networks (CNNs)
* Transfer Learning models
* Vision Transformer (ViT)

The Vision Transformer model was used to improve multi-class disease classification performance and capture complex image patterns from retinal scans.

---

## Performance Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Streamlit Web Application

The project includes a user-friendly web application where users can:

* Upload retinal fundus images
* Predict eye diseases in real-time
* View prediction results instantly

### Run the Application Locally

```bash
streamlit run eye_disease_detection.py
```

---

## Installation

### Clone the Repository

```bash
git clone <repository-link>
cd Eye-Disease-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

### Using Google Colab

1. Open the notebook file
2. Enable GPU runtime
3. Run all cells
4. Train or test the model

The trained `.keras` model files will be saved automatically.

---

## Future Improvements

* Improve model accuracy using advanced architectures
* Add explainable AI (XAI) visualizations
* Deploy using cloud platforms
* Support additional eye diseases
* Optimize inference speed for real-time clinical use

---

## Applications

* Early detection of eye diseases
* AI-assisted medical diagnosis
* Healthcare screening systems
* Clinical decision support systems

---

## Conclusion

This project demonstrates the application of Deep Learning and Computer Vision in the healthcare domain for automated eye disease detection. By combining advanced neural network architectures with an interactive deployment platform, the system provides an efficient and scalable solution for retinal disease classification.

---

## Author

Srijith Chetla
