import cv2
import numpy as np
import os 

X = []
Y = []

TARGET_SIZE = 128 

# For Apple 
apple_path = "data_set/Apple"

print("Starting loop... ")
for filename in os.listdir(apple_path):

    img_path = os.path.join(apple_path, filename)
    print("Trying to open:", img_path)

    img = cv2.imread(img_path)

    if img is None:
        print("Failed to load image")
        continue

    h, w = img.shape[:2]
    print("Height:", h, "Width:", w)
    
    # Scalling
    scale = min(TARGET_SIZE/w,TARGET_SIZE/h)
    new_w = int(w*scale)
    new_h = int(h*scale)

    print("New Size",new_w,new_h)

    # Resizing
    img_resized = cv2.resize(img,(new_w,new_h))
    print("Resized Shape", img_resized.shape)

    #Padding
    pad_w = TARGET_SIZE - new_w
    pad_h = TARGET_SIZE - new_h
    pad_left  = pad_w //2
    pad_right = pad_w - pad_left

    pad_top    = pad_h //2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(img_resized,
                                    pad_top,pad_bottom,
                                    pad_left,pad_right,
                                    cv2.BORDER_CONSTANT,
                                    value=(0,0,0))
    
    print("Final Shape", img_padded.shape)
    print("---------")


    # Gray Scale 
    gray = cv2.cvtColor(img_padded,cv2.COLOR_BGR2GRAY)
    print("Gray Scale: ",gray.shape)

    # Normalizing
    normalized = gray.astype(np.float32)/255.0

    print("Minimum Value :",normalized.min())
    print("Maximum Value :",normalized.max())

    # Flattening
    flattened = normalized.flatten()
    print("Flattened Shape",flattened.shape)

    X.append(flattened)
    Y.append(0) # 0 for apple
   
    print("Total Samples: ",len(X))
    print("Total Labels: ", len(Y))


# Now For Banana 

banana_path = "data_set/Banana"

print("Starting loop... ")
for filename in os.listdir(banana_path):

    img_path = os.path.join(banana_path, filename)
    print("Trying to open:", img_path)

    img = cv2.imread(img_path)

    if img is None:
        print("Failed to load image")
        continue

    h, w = img.shape[:2]
    print("Height:", h, "Width:", w)
    
    # Scalling
    scale = min(TARGET_SIZE/w,TARGET_SIZE/h)
    new_w = int(w*scale)
    new_h = int(h*scale)

    print("New Size",new_w,new_h)

    # Resizing
    img_resized = cv2.resize(img,(new_w,new_h))
    print("Resized Shape", img_resized.shape)

    #Padding
    pad_w = TARGET_SIZE - new_w
    pad_h = TARGET_SIZE - new_h
    pad_left  = pad_w //2
    pad_right = pad_w - pad_left

    pad_top    = pad_h //2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(img_resized,
                                    pad_top,pad_bottom,
                                    pad_left,pad_right,
                                    cv2.BORDER_CONSTANT,
                                    value=(0,0,0))
    
    print("Final Shape", img_padded.shape)
    print("---------")


    # Gray Scale 
    gray = cv2.cvtColor(img_padded,cv2.COLOR_BGR2GRAY)
    print("Gray Scale: ",gray.shape)

    # Normalizing
    normalized = gray.astype(np.float32)/255.0

    print("Minimum Value :",normalized.min())
    print("Maximum Value :",normalized.max())

    # Flattening
    flattened = normalized.flatten()
    print("Flattened Shape",flattened.shape)

    X.append(flattened)
    Y.append(1) # 1 for Banana
   
    print("Total Samples: ", len(X))
    print("Total Labels: " ,  len(Y))