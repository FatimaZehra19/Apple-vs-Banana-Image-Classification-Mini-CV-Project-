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
We compared raw pixel features with HOG features on Apple vs Banana dataset.

## ✅ Results
Training time: 0.11 seconds
Accuracy: 1.0
Confusion Matrix: 
[[63  0]
 [ 0 61]]
 
---- RAW PIXELS ----
Feature length: 16384
Confusion matrix:
 [[63  0]
 [ 0 61]]
Training time: 0.16 seconds
Accuracy: 1.0

---- HOG FEATURES ----
Feature length: 8100
Accuracy: 1.0
Training time: 0.1598665714263916
Confusion matrix:
 [[63  0]
 [ 0 61]]

---
🔎 Observations

- Both Raw Pixel and HOG features achieved perfect classification accuracy (100%) on this dataset.
- HOG reduced feature dimensionality by approximately 50% (16384 → 8100).
- Training time for HOG was slightly higher despite lower dimensionality. This is likely due to:
- Additional feature computation overhead
- Dataset simplicity allowing raw pixels to perform equally well
- The dataset appears to be linearly separable under both representations due to:
   Clean background
   Controlled lighting
   Distinct object shapes

Although accuracy is identical, HOG provides a more structured and illumination-robust representation compared to raw intensity-based features.

---

## 🚀 Future Improvements
- Try SVM classifier
- Test on real-world images
- Extend to multi-class classification
