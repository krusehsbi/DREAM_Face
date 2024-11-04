import os
import pickle

import numpy as np
import cv2


def load_image_as_array(directory, filename):
    image_path = os.path.join(directory, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (128, 128))  # Resize to a fixed size
    return image


def get_subdirectories(directory):
    subdirectories = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            subdirectories.append(item)
    return subdirectories

def load_non_face_data(
        directory,
        images,
        labels,
        deserialize_data,
        serialize_data):
    subdirectories = get_subdirectories(directory)
    # Recursively check if we are in a class directory and if so add the images in that directory
    if len(subdirectories) != 0:
        for subdirectory in subdirectories:
            load_non_face_data(os.path.join(directory, subdirectory), images, labels, deserialize_data, serialize_data)
        # Only load images if this is a class directory
        return

    non_face_images, non_face_labels = load_non_face_dir(directory, deserialize_data, serialize_data)
    append_list(images, non_face_images)
    append_list(labels, non_face_labels)


def load_non_face_dir(directory, deserialize_data, serialize_data):
    class_name = os.path.basename(directory)
    if deserialize_data:
        images, labels = deserialize_saved_data("non_face_" + class_name)
    else:
        images, labels = [], []

    if len(images) == 0 or len(labels) == 0:
        for filename in os.listdir(directory):
            if not filename.endswith('.jpg'):
                continue

            image_array = load_image_as_array(directory, filename)
            if image_array is None:
                continue

            images.append(image_array)
            labels.append([0, 200, 200])  # 0 at the front signals no face present

        if serialize_data:
            serialize_loaded_data(images, labels, "non_face_" + class_name)

    return images, labels

def load_face_data(
        directory,
        images,
        labels,
        deserialize_data,
        serialize_data):
    # Load face images
    for filename in os.listdir(directory):
        if not filename.endswith('.jpg'):
            continue

        # Split the filename into components
        parts = filename.split('_')
        if len(parts) < 4:
            continue

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

        image = load_image_as_array(directory, filename)
        images.append(image)
        labels.append([1, age, gender])  # Label '1' for face images


# Append b to a
def append_list(a, b):
    for i in range(len(b)):
        a.append(b[i])


def load_data(image_directories : [str],
              non_face_directories : [str],
              deserialize_data = True,
              serialize_data = True):
    images = []
    labels = []
    # Load face images
    print("Loading faces")
    for image_directory in image_directories:
        if not os.path.isdir(image_directory):
            print(f'{image_directory} does not exist checking next directory.')
            continue

        load_face_data(image_directory, images, labels, deserialize_data, serialize_data)

    print("Loading non-faces")
    for non_face_directory in non_face_directories:
        if not os.path.isdir(non_face_directory):
            print(f'{non_face_directory} does not exist checking next directory.')
            continue

        load_non_face_data(non_face_directory, images, labels, deserialize_data, serialize_data)

    return np.array(images), np.array(labels)

def serialize_loaded_data(images, labels, name):
    print(f"Serializing Data '{name}' to data/tmp")
    os.makedirs("data/tmp", exist_ok=True)
    with open(f"data/tmp/{name}.pickle", 'ab') as file:
        pickle.dump((images, labels), file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()


def deserialize_saved_data(name):
    print(f"Trying to deserializing Data '{name}' from data/tmp")
    images, labels = [], []
    if os.path.exists(f"data/tmp/{name}.pickle"):
        print(f"Deserializing Data '{name}' from data/tmp")
        with open(f"data/tmp/{name}.pickle", 'rb') as file:
            images, labels = pickle.load(file)
            file.close()
    else:
        print(f"Dataset {name} does not exist and will now be generated.")
    return images, labels

def load_data_old(face_directory, non_face_directory):
    images = []
    labels = []

    load_face_data(face_directory, images, labels, False, False)

    # Load non-face images
    for class_name in os.listdir(non_face_directory):
        class_path = os.path.join(non_face_directory, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.endswith('.jpg'):
                image = load_image_as_array(class_path, filename)
                images.append(image)
                labels.append([0, 200, 2])  # Label '0' for non-face images, no age/gender

    return np.array(images), np.array(labels)
