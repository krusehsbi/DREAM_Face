import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import saving, ops
import FaceIdentifier


# Function to load a specified number of random images with true labels
def load_random_images(face_dir, non_face_dir, num_samples=6):
    images = []
    true_labels = []

    # Load random face images
    face_files = random.sample(os.listdir(face_dir), num_samples // 2)
    for filename in face_files:
        parts = filename.split('_')
        if len(parts) >= 3:
            age = int(parts[0])
            gender = int(parts[1])
            face_label = 1  # Face
            image_path = os.path.join(face_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            images.append(image)
            true_labels.append([face_label, age, gender])

    # Load random non-face images
    non_face_class = random.choice(os.listdir(non_face_dir))
    if os.path.isdir(non_face_class):
        non_face_path = os.path.join(non_face_dir, non_face_class)
        non_face_files = random.sample(os.listdir(non_face_path), num_samples // 2)
    else:
        non_face_path = non_face_dir
        non_face_files = random.sample(os.listdir(non_face_dir), num_samples // 2)

    for filename in non_face_files:
        face_label = 0  # Non-face
        age = 200  # Placeholder for age in non-face images
        gender = 2  # Placeholder for gender in non-face images
        image_path = os.path.join(non_face_path, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        images.append(image)
        true_labels.append([face_label, age, gender])

    return np.array(images), np.array(true_labels)


# Function to show images with predictions
def show_random_predictions(images, true_labels, save_path='predictions.png'):
    predictions = model.predict(images)
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        image = images[i]
        true_face, true_age, true_gender = true_labels[i]

        # Extract single values from the predictions
        pred_face = float(predictions['face_output'][i])  # Access the first element to get the scalar
        pred_age = round(predictions['age_output'][i][0])  # Access the first element for the scalar
        pred_gender = float(ops.argmax(predictions['gender_output'][i]))  # Argmax works directly as expected

        # Display the image with true vs predicted labels
        plt.subplot(2, 3, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis('off')
        title_true = f"True: Face 100%, Age {true_age}, Gender {"male" if true_gender == 0 else "female" }\n" if true_face == 1 else "True: Face 0%\n"
        title_pred = f"Pred: Face {100 * pred_face:.2f}%, Age {pred_age}, Gender {"male" if pred_gender == 0 else "female" }" if pred_face >= 0.5 else f"Pred: Face {100 * pred_face:.2f}%"
        plt.title(title_true + title_pred)

    plt.tight_layout()
    # Save plot to file instead of showing it
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == '__main__':
    # Load model
    model = saving.load_model('saved_models/Face.keras')

    # Directories
    face_directory = 'data/utk-face'
    non_face_directory = 'data/nonface/openimages'

    # Load random images and display predictions
    images, true_labels = load_random_images(face_directory, non_face_directory)
    show_random_predictions(images, true_labels)
