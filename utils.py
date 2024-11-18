import math
import os
import pickle

import numpy as np
from keras import utils, applications
import matplotlib.pyplot as plt
import csv

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

    load_non_face_dir(directory, images, labels, deserialize_data, serialize_data)


def load_non_face_dir(directory, images, labels, deserialize_data, serialize_data):
    class_name = os.path.basename(directory)
    if deserialize_data:
        images_temp, labels_temp = deserialize_saved_data("non_face_" + class_name)
    else:
        images_temp, labels_temp = [], []

    if len(images_temp) == 0 or len(labels_temp) == 0:
        # Load non-face images
        for filename in os.listdir(directory):
            if not filename.endswith('.jpg'):
                continue

            image = load_image_as_array(directory, filename)
            images_temp.append(image)
            labels_temp.append([0, 200, 2])  # Label '0' for non-face images, no age/gender

        if serialize_data:
            serialize_loaded_data(images_temp, labels_temp, "non_face_" + class_name)

    append_list(images, images_temp)
    append_list(labels, labels_temp)

def load_face_data(
        directory,
        images,
        labels,
        deserialize_data,
        serialize_data
):
    if deserialize_data:
        images_temp, labels_temp = deserialize_saved_data("face")
    else:
        images_temp, labels_temp = [], []

    if len(images_temp) == 0 or len(labels_temp) == 0:
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
            images_temp.append(image)
            labels_temp.append([1, age, gender])  # Label '1' for face images

        if serialize_data:
            serialize_loaded_data(images_temp, labels_temp, "face")

    append_list(images, images_temp), append_list(labels, labels_temp)


# Append b to a
def append_list(a, b):
    for i in range(len(b)):
        a.append(b[i])

def serialize_loaded_data(images, labels, name):
    print(f"Serializing Data '{name}' to data/tmp")

    if len(images) == 0 or len(labels) == 0:
        print(f"{name} does not contain any images or labels to serialize and will be skipped.")
        return

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

def load_data(
        face_directories,
        non_face_directories,
        preprocess_fnc,
        deserialize_data = True,
        serialize_data = True,
):
    images = []
    labels = []

    # Load face images
    print("Loading faces")
    for face_directory in face_directories:
        if not os.path.isdir(face_directory):
            print(f'{face_directory} does not exist checking next directory.')
            continue
        load_face_data(face_directory, images, labels, deserialize_data, serialize_data)

    print("Loading non-faces")
    for non_face_directory in non_face_directories:
        if not os.path.isdir(non_face_directory):
            print(f'{non_face_directory} does not exist checking next directory.')
            continue

        load_non_face_data(non_face_directory, images, labels, deserialize_data, serialize_data)

    images, labels = np.array(images), np.array(labels)
    if preprocess_fnc is not None:
        images = preprocess_fnc(images)

    return images, labels

def shuffle_arrays(array1, array2):
    assert len(array1) == len(array2)
    permutation = np.random.permutation(len(array1))
    return array1[permutation], array2[permutation]

class DataGeneratorIdentifier(utils.Sequence):
    def __init__(self, images, labels_face, labels_age, labels_gender, batch_size, shuffle=True):
        super(DataGeneratorIdentifier, self).__init__()
        self.images = images
        self.labels_face = labels_face
        self.labels_age = labels_age
        self.labels_gender = labels_gender
        self.batch_size = batch_size
        self.datalen = len(images)
        self.indices = np.arange(self.datalen)
        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1)*self.batch_size]
        batch_x = self.images[batch_indices]
        batch_y_face = self.labels_face[batch_indices]
        batch_y_age = self.labels_age[batch_indices]
        batch_y_gender = self.labels_gender[batch_indices]

        return (
            batch_x,
            {'face_output': batch_y_face, 'age_output': batch_y_age, 'gender_output': batch_y_gender})

class DataGeneratorDetector(utils.Sequence):
    def __init__(self, images, labels_face, batch_size, shuffle=True):
        super(DataGeneratorDetector, self).__init__()
        self.images = images
        self.labels_face = labels_face
        self.batch_size = batch_size
        self.datalen = len(images)
        self.indices = np.arange(self.datalen)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1)*self.batch_size]
        batch_x = self.images[batch_indices]
        batch_y_face = self.labels_face[batch_indices]

        return batch_x, batch_y_face

def PlotHistory(history, direct=False):
    if not direct:
        history_dict = history.history
    else:
        history_dict = history
    # Get the number of subplots needed based on the keys
    categories = [key for key in history_dict.keys() if not key.startswith('val_') and not key.startswith('learning_rate')]
    num_categories = len(categories)

    # Calculate the grid size to make it as close to a square as possible
    num_cols = math.ceil(math.sqrt(num_categories))
    num_rows = math.ceil(num_categories / num_cols)

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Flatten the axes array if it's more than one row or column
    axes = axes.flatten() if num_categories > 1 else [axes]

    # Plot each category in its subplot
    for idx, category in enumerate(categories):
        ax = axes[idx]
        ax.plot(history_dict[category])
        ax.plot(history_dict['val_' + category])
        ax.set_title(category)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['train', 'val'], loc='upper left')

    # Hide any unused subplots
    for idx in range(num_categories, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout
    plt.tight_layout()
    plt.show()

def PlotExportedHistory(filename):
    # Initialize the dictionary to hold the history data
    reconstructed_history = {}

    # Read the CSV file
    with open(filename, mode='r') as file:
        reader = csv.reader(file)

        # Read the header row to get the keys
        headers = next(reader)

        # Initialize lists in the dictionary for each key
        for header in headers:
            reconstructed_history[header] = []

        # Read the data rows and append to corresponding lists in the dictionary
        for row in reader:
            for idx, value in enumerate(row):
                reconstructed_history[headers[idx]].append(float(value))

    PlotHistory(reconstructed_history, True)