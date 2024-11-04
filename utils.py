import os
import cv2
import numpy as np

def load_data(face_directory, non_face_directory):
    images = []
    labels = []

    # Load face images
    for filename in os.listdir(face_directory):
        if filename.endswith('.jpg'):
            parts = filename.split('_')
            if len(parts) < 4:
                continue

            age = int(parts[0])  # Convert age to int
            gender = int(parts[1])  # Convert gender to int
            race = int(parts[2])  # Convert race to int

            image_path = os.path.join(face_directory, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = cv2.resize(image, (128, 128))  # Resize to a fixed size
            images.append(image)
            labels.append([1, age, gender])  # Label '1' for face images

    # Load non-face images
    for class_name in os.listdir(non_face_directory):
        class_path = os.path.join(non_face_directory, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append([0, 200, 2])  # Label '0' for non-face images, no age/gender

    return np.array(images), np.array(labels)
