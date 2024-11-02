import os

from keras import utils
import numpy as np

def load_data(image_directory, non_face_directory):
    images = []
    labels = []

    # Load face images
    for filename in os.listdir(image_directory):
        if not filename.endswith('.jpg'):
            continue

        # Split the filename into components
        parts = filename.split('_')
        if len(parts) < 4:
            continue

        age = None
        gender = None
        # Extract labels from filename
        try:
            age = int(parts[0])  # Convert age to int
        except ValueError:
            print(f"Age {parts[0]} is not a valid number. File '{filename}'")
            continue

        try:
            gender = int(parts[1])  # Convert gender to int
        except ValueError:
            print(f"Gender {parts[1]} is not a valid number. File '{filename}'")
            continue

        image = utils.load_img(
            path=os.path.join(image_directory, filename),
            color_mode="rgb",
            target_size=(128, 128),
            interpolation="bilinear",
            keep_aspect_ratio=False
        )

        if image is None:
            print(f"Image {filename} was not loaded.")
            continue

        image_array = utils.img_to_array(image)
        images.append(image_array)
        labels.append([1, age, gender]) # 1 at the front signals face present

    # Load non-face images
    for class_name in os.listdir(non_face_directory):
        class_path = os.path.join(non_face_directory, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if not filename.endswith('.jpg'):
                continue

            image = utils.load_img(
                path=os.path.join(image_directory, filename),
                color_mode="rgb",
                target_size=(128, 128),
                interpolation="bilinear",
                keep_aspect_ratio=False
            )

            if image is None:
                print(f"Image {filename} was not loaded.")
                continue

            image_array = utils.img_to_array(image)
            images.append(image_array)
            labels.append([0, -1, -1]) # 0 at the front signals no face present

    return np.array(images), np.array(labels)
