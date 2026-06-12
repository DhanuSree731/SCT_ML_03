# 🐱🐶 Cats vs Dogs Image Classifier

A Machine Learning web application built using **Support Vector Machine (SVM)** and **HOG (Histogram of Oriented Gradients)** feature extraction to classify images as either a Cat or a Dog.

This project was developed as part of the **SkillCraft Technology Machine Learning Internship (Task 3)**.

---

## 📌 Project Overview

The Cats vs Dogs Classifier is an image classification system that automatically identifies whether an uploaded image contains a **cat or a dog** using traditional machine learning techniques.

The project is implemented in two phases:

---

## 📓 Phase 1: Jupyter Notebook

The complete machine learning pipeline was developed and trained in Jupyter Notebook.

### 🔹 Steps Performed:
- Dataset loading and preprocessing  
- Image resizing and grayscale conversion  
- Feature extraction using HOG descriptors  
- Feature scaling using StandardScaler  
- SVM model training  
- Model evaluation and testing  
- Prediction on sample images  

---

## 🌐 Phase 2: Streamlit Web Application

The trained SVM model was deployed into an interactive Streamlit web application.

### 🔹 Application Features:
- Image upload functionality  
- Real-time image classification (Cat vs Dog)  
- Prediction output display  
- Confidence score visualization  
- Probability distribution chart  
- Clean and interactive dashboard UI  

---

## 🚀 Features

### 📸 Image Upload
- Upload JPG / PNG images  
- Preview uploaded image instantly  

### 🤖 Prediction System
- Classifies image as Cat or Dog  
- Uses trained SVM model  

### 📊 Confidence Score
- Displays prediction confidence percentage  
- Progress bar visualization  

### 📈 Probability Analysis
- Bar chart showing class probabilities  
- Visual comparison between Cat vs Dog  

### 🌐 Interactive Dashboard
- Streamlit-based UI  
- Sidebar navigation  
- Clean and responsive layout  

---

## 🛠️ Technologies Used

- **Python**  
- **OpenCV**  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **Scikit-Learn**  
- **Scikit-Image (HOG Features)**  
- **Streamlit**  

---

## 🤖 Machine Learning Model

### 🔹 Algorithm Used:
- Support Vector Machine (SVM)

### 🔹 Feature Extraction:
- HOG (Histogram of Oriented Gradients)

### 🔹 Workflow:
1. Image Input  
2. Preprocessing (Resize + Grayscale)  
3. Feature Extraction (HOG)  
4. Feature Scaling  
5. SVM Prediction  
6. Result Display  

---

## 📂 Dataset Pipeline

- Cat Images  
- Dog Images  
- Image Preprocessing  
- HOG Feature Extraction  
- Model Training & Testing  

---

## 📸 Application Pages

### 🐾 Dashboard
- Project overview  
- Tech stack display  
- Model summary metrics  
- Clean UI banner  

### 🎯 Prediction Page
- Upload image  
- View prediction result  
- Confidence score  
- Probability graph  

### 📚 About Project
- ML workflow explanation  
- Dataset details  
- Future improvements  

---

## ▶️ How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/DhanuSree731/SCT_ML_03.git
2️⃣ Move to Project Folder
cd SCT_ML_03
3️⃣ Install Requirements
pip install -r requirements.txt
4️⃣ Run Streamlit App
streamlit run app.py
📊 Learning Outcomes

This project helped in understanding:

Image Processing Techniques
Feature Extraction using HOG
Support Vector Machine (SVM)
Computer Vision Basics
Model Training & Evaluation
Streamlit Web App Development
End-to-End ML Deployment
🔮 Future Enhancements
🤖 Deep Learning Upgrade
CNN-based classification
Transfer learning (VGG16 / ResNet)
📷 Advanced Features
Webcam-based real-time detection
Multi-class animal classification
Image augmentation
☁️ Deployment
Cloud deployment (Streamlit Cloud / AWS)
API integration
📊 UI Improvements
Plotly interactive charts
Dark mode support
Mobile-friendly design
🎯 Internship Task

SkillCraft Technology – Machine Learning Internship
Task 3: Image Classification using SVM

👩‍💻 Developer

Dhanu Sree
Machine Learning Intern – SkillCraft Technology

🔗 GitHub: DhanuSree731