import os
import pickle

from keras import utils
import numpy as np


def load_image_as_array(directory, filename):
    image = utils.load_img(
        path=os.path.join(directory, filename),
        color_mode="rgb",
        target_size=(128, 128),
        interpolation="bilinear",
        keep_aspect_ratio=False
    )

    if image is None:
        print(f"Image {filename} was not loaded.")
        return None

    image_array = utils.img_to_array(image, dtype=np.uint8)
    image.close()
    return image_array


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

    if deserialize_data:
        images_temp, labels_temp = deserialize_saved_data("face")
    else:
        images_temp, labels_temp = [], []

    if len(images_temp) == 0 or len(labels_temp) == 0:
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

            image_array = load_image_as_array(directory, filename)
            if image_array is None:
                continue
            images_temp.append(image_array)
            labels_temp.append([1, age, gender])  # 1 at the front signals face present

        if serialize_data:
            serialize_loaded_data(images_temp, labels_temp, "face")

    append_list(images, images_temp), append_list(labels, labels_temp)

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