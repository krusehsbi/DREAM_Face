import os
from keras import utils
import numpy as np

def load_data(image_directory, non_face_directory):
    images = []
    labels = []

    print("Images faces")
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

    print("Images imagenet")
    if non_face_directory is not None:
        # Load non-face images
        for class_name in os.listdir(non_face_directory):
            class_path = os.path.join(non_face_directory, class_name)
            if not os.path.isdir(class_path):
                continue

            for filename in os.listdir(class_path):
                if not filename.endswith('.jpg'):
                    continue

                image = utils.load_img(
                    path=os.path.join(class_path, filename),
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
                labels.append([0, 200, 200]) # 0 at the front signals no face present

    return np.array(images), np.array(labels)

def serialize_loaded_data(images, labels):
    print("Serializing Data to data/tmp")
    os.makedirs("data/tmp", exist_ok=True)
    np.save("data/tmp/images.npy", images)
    np.save("data/tmp/labels.npy", labels)

def deserialize_saved_data():
    print("Deserializing Data from data/tmp")
    images, labels = None, None
    if os.path.exists("data/tmp/images.npy") and os.path.exists("data/tmp/labels.npy"):
        images = np.load("data/tmp/images.npy")
        labels = np.load("data/tmp/labels.npy")
    return images, labels