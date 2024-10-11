import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(image_directory):
    images = []
    labels = []
    
    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg'):
            # Split the filename into components
            parts = filename.split('_')
            if len(parts) < 4:
                continue
            
            # Extract labels from filename
            age = int(parts[0])  # Convert age to int
            gender = int(parts[1])  # Convert gender to int
            race = int(parts[2])  # Convert race to int
            
            # Load the image
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = cv2.resize(image, (128, 128))  # Resize to a fixed size
            images.append(image)
            labels.append([age, gender, race])
    
    return np.array(images), np.array(labels)