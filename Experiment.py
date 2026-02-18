from Preprocessing import process_single_image
from Features import extract_raw, extract_hog
from Train import train_model
import os

images = []
labels = []

# Example dataset loading loop
for label, folder in enumerate(["data_set/Apple", "data_set/Banana"]):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        
        # Preprocess
        img = process_single_image(path)
        
        images.append(img)
        labels.append(label)

# -------- RAW Experiment --------
X_raw = [extract_raw(img) for img in images]
acc_raw, cm_raw, time_raw = train_model(X_raw, labels)

print("---- RAW PIXELS ----")
print("Feature length:", len(X_raw[0]))
print("Accuracy:", acc_raw)
print("Training time:", time_raw)
print("Confusion matrix:\n", cm_raw)


# -------- HOG Experiment --------
X_hog = [extract_hog(img) for img in images]
acc_hog, cm_hog, time_hog = train_model(X_hog, labels)

print("\n---- HOG FEATURES ----")
print("Feature length:", len(X_hog[0]))
print("Accuracy:", acc_hog)
print("Training time:", time_hog)
print("Confusion matrix:\n", cm_hog)
