import math
import os
import pickle

import numpy as np
from keras import utils, applications
import matplotlib.pyplot as plt
import csv


def load_image_as_array(directory, filename):
    """
    Loads an image from a specified directory and converts it to a NumPy array.

    Parameters:
    - directory: str, path to the directory containing the image.
    - filename: str, name of the image file.

    Returns:
    - image_array: NumPy array representing the image (RGB, 128x128).
    """
    image = utils.load_img(
        path=os.path.join(directory, filename),
        color_mode="rgb",  # Load the image in RGB mode
        target_size=(128, 128),  # Resize the image to 128x128
        interpolation="bilinear",  # Bilinear interpolation for resizing
        keep_aspect_ratio=False  # Do not maintain the aspect ratio
    )

    if image is None:
        print(f"Image {filename} was not loaded.")
        return None

    # Convert the loaded image to a NumPy array
    image_array = utils.img_to_array(image, dtype=np.uint8)
    image.close()  # Close the image to free resources
    return image_array


def get_subdirectories(directory):
    """
    Returns a list of subdirectories in a given directory.

    Parameters:
    - directory: str, path to the directory to be scanned.

    Returns:
    - subdirectories: list of str, names of subdirectories.
    """
    subdirectories = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            subdirectories.append(item)
    return subdirectories


def load_non_face_data(directory, images, labels, deserialize_data, serialize_data):
    """
    Recursively loads non-face images from a directory structure, adding them to the provided lists.

    Parameters:
    - directory: str, the root directory to load non-face images from.
    - images: list, the list to append image arrays to.
    - labels: list, the list to append image labels to.
    - deserialize_data: bool, if True, attempts to load preprocessed data from a serialized format.
    - serialize_data: bool, if True, saves the loaded data for future use.
    """
    subdirectories = get_subdirectories(directory)

    # If subdirectories exist, recurse into them
    if len(subdirectories) != 0:
        for subdirectory in subdirectories:
            load_non_face_data(os.path.join(directory, subdirectory), images, labels, deserialize_data, serialize_data)
        return

    # If no subdirectories, load images from the current directory
    load_non_face_dir(directory, images, labels, deserialize_data, serialize_data)


def load_non_face_dir(directory, images, labels, deserialize_data, serialize_data):
    """
    Loads non-face images from a directory and assigns a label to each image.

    Parameters:
    - directory: str, the directory to load non-face images from.
    - images: list, the list to append image arrays to.
    - labels: list, the list to append image labels to.
    - deserialize_data: bool, if True, attempts to load preprocessed data from a serialized format.
    - serialize_data: bool, if True, saves the loaded data for future use.
    """
    class_name = os.path.basename(directory)

    # Attempt to load previously serialized data
    if deserialize_data:
        images_temp, labels_temp = deserialize_saved_data("non_face_" + class_name)
    else:
        images_temp, labels_temp = [], []

    if len(images_temp) == 0 or len(labels_temp) == 0:
        # If no serialized data, load images manually
        for filename in os.listdir(directory):
            if not filename.endswith('.jpg'):
                continue

            image = load_image_as_array(directory, filename)
            images_temp.append(image)
            labels_temp.append([0, 200, 2])  # Label '0' for non-face images, no specific age/gender

        # Serialize the loaded data for future use
        if serialize_data:
            serialize_loaded_data(images_temp, labels_temp, "non_face_" + class_name)

    append_list(images, images_temp)
    append_list(labels, labels_temp)


def load_face_data(directory, images, labels, deserialize_data, serialize_data):
    """
    Loads face images from a directory, extracting labels from filenames.

    Parameters:
    - directory: str, the directory to load face images from.
    - images: list, the list to append image arrays to.
    - labels: list, the list to append image labels to.
    - deserialize_data: bool, if True, attempts to load preprocessed data from a serialized format.
    - serialize_data: bool, if True, saves the loaded data for future use.
    """
    if deserialize_data:
        images_temp, labels_temp = deserialize_saved_data("face")
    else:
        images_temp, labels_temp = [], []

    if len(images_temp) == 0 or len(labels_temp) == 0:
        # If no serialized data, load images manually
        for filename in os.listdir(directory):
            if not filename.endswith('.jpg'):
                continue

            # Split the filename to extract age and gender
            parts = filename.split('_')
            if len(parts) < 4:
                continue

            # Extract age and gender from filename
            try:
                age = int(parts[0])
            except ValueError:
                print(f"Age {parts[0]} is not a valid number. File '{filename}'")
                continue

            try:
                gender = int(parts[1])
            except ValueError:
                print(f"Gender {parts[1]} is not a valid number. File '{filename}'")
                continue

            image = load_image_as_array(directory, filename)
            images_temp.append(image)
            labels_temp.append([1, age, gender])  # Label '1' for face images

        # Serialize the loaded data for future use
        if serialize_data:
            serialize_loaded_data(images_temp, labels_temp, "face")

    append_list(images, images_temp)
    append_list(labels, labels_temp)


def append_list(a, b):
    """
    Appends the contents of list `b` to list `a`.

    Parameters:
    - a: list, the target list to be appended to.
    - b: list, the list to append.
    """
    for i in range(len(b)):
        a.append(b[i])


def serialize_loaded_data(images, labels, name):
    """
    Serializes the provided image and label data to a file.

    Parameters:
    - images: list, the images to be serialized.
    - labels: list, the labels to be serialized.
    - name: str, the name of the serialized file.
    """
    print(f"Serializing Data '{name}' to data/tmp")

    if len(images) == 0 or len(labels) == 0:
        print(f"{name} does not contain any images or labels to serialize and will be skipped.")
        return

    os.makedirs("data/tmp", exist_ok=True)
    with open(f"data/tmp/{name}.pickle", 'ab') as file:
        pickle.dump((images, labels), file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()


def deserialize_saved_data(name):
    """
    Deserializes image and label data from a file.

    Parameters:
    - name: str, the name of the file to deserialize.

    Returns:
    - images: list, the deserialized images.
    - labels: list, the deserialized labels.
    """
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


def load_data(face_directories, non_face_directories, preprocess_fnc, deserialize_data=True, serialize_data=True):
    """
    Loads both face and non-face images from specified directories, with optional preprocessing.

    Parameters:
    - face_directories: list of str, directories containing face images.
    - non_face_directories: list of str, directories containing non-face images.
    - preprocess_fnc: function, optional preprocessing function to apply to the images.
    - deserialize_data: bool, if True, attempts to load preprocessed data from a serialized format.
    - serialize_data: bool, if True, saves the loaded data for future use.

    Returns:
    - images: NumPy array, preprocessed image data.
    - labels: NumPy array, corresponding labels.
    """
    images = []
    labels = []

    # Load face images
    print("Loading faces")
    for face_directory in face_directories:
        if not os.path.isdir(face_directory):
            print(f'{face_directory} does not exist checking next directory.')
            continue
        load_face_data(face_directory, images, labels, deserialize_data, serialize_data)

    # Load non-face images
    print("Loading non-faces")
    for non_face_directory in non_face_directories:
        if not os.path.isdir(non_face_directory):
            print(f'{non_face_directory} does not exist checking next directory.')
            continue

        load_non_face_data(non_face_directory, images, labels, deserialize_data, serialize_data)

    # Convert lists to NumPy arrays
    images, labels = np.array(images), np.array(labels)
    if preprocess_fnc is not None:
        images = preprocess_fnc(images)

    return images, labels


def shuffle_arrays(array1, array2):
    """
    Shuffles two arrays in unison, maintaining correspondence between them.

    Parameters:
    - array1: NumPy array, the first array to shuffle.
    - array2: NumPy array, the second array to shuffle (must be the same length as array1).

    Returns:
    - shuffled array1, shuffled array2.
    """
    assert len(array1) == len(array2)
    permutation = np.random.permutation(len(array1))
    return array1[permutation], array2[permutation]


class DataGeneratorIdentifier(utils.Sequence):
    """
    Custom data generator for multi-output neural network training (face, age, and gender).
    Inherits from Keras' Sequence class for efficient data loading.
    """

    def __init__(self, images, labels_face, labels_age, labels_gender, batch_size, shuffle=True):
        """
        Initializes the data generator.

        Parameters:
        - images: NumPy array, image data.
        - labels_face: NumPy array, binary labels indicating face presence.
        - labels_age: NumPy array, labels for age classification.
        - labels_gender: NumPy array, labels for gender classification.
        - batch_size: int, the number of samples per batch.
        - shuffle: bool, if True, shuffles the data.
        """
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
        """
        Returns the number of batches per epoch.
        """
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        """
        Generates one batch of data.

        Parameters:
        - idx: int, index of the batch.

        Returns:
        - Tuple of (batch_x, batch_y) where batch_x is the image data and batch_y is a dictionary of labels.
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.images[batch_indices]
        batch_y_face = self.labels_face[batch_indices]
        batch_y_age = self.labels_age[batch_indices]
        batch_y_gender = self.labels_gender[batch_indices]

        return (
            batch_x,
            {'face_output': batch_y_face, 'age_output': batch_y_age, 'gender_output': batch_y_gender}
        )


class DataGeneratorDetector(utils.Sequence):
    """
    Custom data generator for single-output neural network training (face detection).
    Inherits from Keras' Sequence class for efficient data loading.
    """

    def __init__(self, images, labels_face, batch_size, shuffle=True):
        """
        Initializes the data generator.

        Parameters:
        - images: NumPy array, image data.
        - labels_face: NumPy array, binary labels indicating face presence.
        - batch_size: int, the number of samples per batch.
        - shuffle: bool, if True, shuffles the data at the end of each epoch.
        """
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
        """
        Returns the number of batches per epoch.
        """
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        """
        Generates one batch of data.

        Parameters:
        - idx: int, index of the batch.

        Returns:
        - Tuple of (batch_x, batch_y_face) where batch_x is the image data and batch_y_face are the face labels.
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.images[batch_indices]
        batch_y_face = self.labels_face[batch_indices]

        return batch_x, batch_y_face


def PlotHistory(history, direct=False):
    """
    Plots the training and validation loss for each metric stored in the training history.

    Parameters:
    - history: Keras History object or a dictionary containing training metrics.
    - direct: bool, if True, treats the 'history' parameter as a direct dictionary.
    """
    if not direct:
        history_dict = history.history
    else:
        history_dict = history

    # Get the number of subplots needed based on the keys
    categories = [key for key in history_dict.keys() if
                  not key.startswith('val_') and not key.startswith('learning_rate')]
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
    """
    Loads training history data from a CSV file and plots it.

    Parameters:
    - filename: str, path to the CSV file containing training history.
    """
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
