# Apple-vs-Banana-Image-Classification-Mini-CV-Project-
This is my first structured Computer Vision mini project.


## 📌 Overview
This project demonstrates a complete computer vision pipeline using
traditional machine learning techniques.

The goal is to classify images of apples and bananas using image
preprocessing and Logistic Regression.

---

## 🧠 Pipeline
1. Load raw images
2. Resize with aspect ratio preservation
3. Pad images to 128×128
4. Convert to grayscale
5. Normalize pixel values
6. Flatten images into feature vectors
7. Train Logistic Regression model
8. Evaluate using accuracy and confusion matrix

---

## 🛠 Tools & Libraries
- Python
- OpenCV
- NumPy
- Scikit-learn

---

## 📊 Dataset
- Apple images: 310
- Banana images: 310
- Dataset not included in repository (ignored via `.gitignore`)

---

## ✅ Results
- Accuracy: **100%**
- Confusion Matrix:
[[63 0]
[ 0 61]]


---

## 🚀 Future Improvements
- Use HOG features
- Try SVM classifier
- Test on real-world images
- Extend to multi-class classification
