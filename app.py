import streamlit as st
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import hog
from PIL import Image
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("models/cats_dogs_svm.pkl")
scaler = joblib.load("models/scaler.pkl")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🐾 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Prediction",
        "About Project"
    ]
)

# ==================================================
# DASHBOARD
# ==================================================
if page == "Dashboard":

    st.title("🐱🐶 Cats vs Dogs Classifier")

    st.caption(
        "SkillCraft Technology Machine Learning Internship Project"
    )

    st.markdown(
        "### Machine Learning Based Animal Image Classification"
    )

    # Banner Image
    try:
        banner = Image.open("Cats vs Dogs Classifier.jpg")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(
                banner,
                width=600
            )

    except:
        st.warning(
            "Dashboard banner image not found. Place 'cat_dog_banner.jpg' in project folder."
        )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model", "SVM")
    c2.metric("Feature Extraction", "HOG")
    c3.metric("Classes", "2")
    c4.metric("Output", "Cat / Dog")

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("📋 Project Overview")

        st.info("""
        This project classifies images into:

        • Cat 🐱

        • Dog 🐶

        Workflow:

        1. Image Upload

        2. Image Preprocessing

        3. HOG Feature Extraction

        4. SVM Classification

        5. Prediction Output
        """)

    with right:

        st.subheader("🛠 Technology Stack")

        st.success("""
        • Python

        • OpenCV

        • Scikit-Learn

        • HOG Features

        • Support Vector Machine (SVM)

        • Streamlit
        """)

    st.markdown("---")

    st.subheader("🎯 Project Goal")

    st.write("""
    The objective of this project is to automatically classify
    uploaded animal images as either a Cat or a Dog using
    HOG feature extraction and a Support Vector Machine (SVM)
    classifier.
    """)

# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Prediction":

    st.title("🐾 Image Classification")

    uploaded_file = st.file_uploader(
        "Upload Cat or Dog Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            rgb,
            caption="Uploaded Image",
            use_container_width=True
        )

        img = cv2.resize(
            image,
            (128, 128)
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        features = hog(
            gray,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        features = scaler.transform(
            [features]
        )

        prediction = model.predict(
            features
        )[0]

        probability = model.predict_proba(
            features
        )[0]

        confidence = max(probability) * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 0:

            st.success("🐱 Cat")

        else:

            st.success("🐶 Dog")

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(confidence / 100, 1.0)
        )

        st.markdown("---")

        st.subheader("Prediction Probabilities")

        result_df = pd.DataFrame({

            "Class": ["Cat", "Dog"],

            "Probability": [
                probability[0] * 100,
                probability[1] * 100
            ]
        })

        fig, ax = plt.subplots(figsize=(6,4))

        ax.bar(
            result_df["Class"],
            result_df["Probability"]
        )

        ax.set_ylabel("Probability (%)")

        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

# ==================================================
# ABOUT PROJECT
# ==================================================

else:

    st.title("📚 Project Information")

    st.subheader("Project Objective")

    st.write("""
    Build a machine learning model capable of
    classifying images as Cats or Dogs.
    """)

    st.subheader("Dataset")

    st.write("""
    • Cat Images

    • Dog Images

    • Image Preprocessing

    • HOG Feature Extraction
    """)

    st.subheader("Machine Learning Pipeline")

    pipeline = pd.DataFrame({

        "Step": [
            "Data Collection",
            "Image Resize",
            "Gray Conversion",
            "HOG Features",
            "Scaling",
            "SVM Training",
            "Prediction"
        ]
    })

    st.table(pipeline)

    st.subheader("Future Enhancements")

    st.write("""
    • Larger Dataset

    • CNN Deep Learning Models

    • Webcam Detection

    • Multi-Class Animal Classification

    • Cloud Deployment
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Developed by Dhanu Sree | SkillCraft Technology ML Internship"
)