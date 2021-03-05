# Importing Libraries
import numpy as np
import os
from pathlib import Path
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Dataset Preparation
p = Path("Dataset/images/")
# print(type(p))

dirs = p.glob("*")

labels_dict = {"cat":0, "dog":1, "horse":2, "human":3}
image_data = []
labels = []

for folder_dir in dirs:
    # print(folder_name)
    label = str(folder_dir).split("\\")[-1][:-1]
    # print(label)
    
    for img_path in folder_dir.glob("*.jpg"):
        img = image.load_img(img_path, target_size=(32,32))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[label])
        
# convert this into numpy array
image_data = np.array(image_data, dtype="float32")/255.0
labels = np.array(labels)
        