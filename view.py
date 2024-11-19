import argparse
import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import saving, ops, applications
from utils import load_image_as_array
import FaceIdentifier
import FaceDetector

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
            images.append(load_image_as_array(face_dir, filename))
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
        images.append(load_image_as_array(non_face_dir, filename))
        true_labels.append([face_label, age, gender])

    return np.array(images), np.array(true_labels)


# Function to show images with face, age and gender predictions
def show_random_predictions_face(images, true_labels, noshow, save_path='predictions_face.png'):
    predictions = model.predict(applications.efficientnet.preprocess_input(images))
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        image = images[i]
        true_face, true_age, true_gender = true_labels[i]

        # Extract single values from the predictions
        pred_face = float(predictions['face_output'][i][0])  # Access the first element to get the scalar
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
    # Save plot to file and display it
    if not noshow:
        plt.show()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# Function to show images with face predictions
def show_random_predictions_face_detector(images, true_labels, noshow, save_path='predictions_face_detector.png'):
    predictions = model.predict(applications.efficientnet.preprocess_input(images))
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        image = images[i]
        true_face, true_age, true_gender = true_labels[i]

        # Extract single values from the predictions
        pred_face = float(ops.sigmoid(predictions[i][0]))  # Access the first element to get the scalar

        # Display the image with true vs predicted labels
        plt.subplot(2, 3, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis('off')
        title_true = f"True: Face {"100%" if true_face == 1 else "0%"}"
        title_pred = f"Pred: Face {100 * pred_face:.2f}%"
        plt.title(title_true + title_pred)

    plt.tight_layout()
    # Save plot to file and display it
    if not noshow:
        plt.show()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def load_all_images(dir):
    images = []
    for filename in os.listdir(dir):
        images.append(load_image_as_array(dir, filename))

    return np.array(images)

def predict_all_images(images, model, noshow, save_path='all_predictions.png'):
    predictions = model.predict(applications.efficientnet.preprocess_input(images))
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        image = images[i]

        # Extract single values from the predictions
        pred_face = float(predictions['face_output'][i][0])  # Access the first element to get the scalar
        pred_age = round(predictions['age_output'][i][0])  # Access the first element for the scalar
        pred_gender = float(ops.argmax(predictions['gender_output'][i]))  # Argmax works directly as expected


        # Calculate the grid size to make it as close to a square as possible
        num_cols = math.ceil(math.sqrt(len(images)))
        num_rows = math.ceil(len(images) / num_cols)

        # Display the image with predicted labels
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis('off')
        title_pred = f"Pred: Face {100 * pred_face:.2f}%, Age {pred_age}, Gender {"male" if pred_gender == 0 else "female"}" if pred_face >= 0.5 else f"Pred: Face {100 * pred_face:.2f}%"
        plt.title(title_pred)

    plt.tight_layout()
    if not noshow:
        plt.show()
    # Save plot to file
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Model-Viewer',
        description='Can be used to view the results of a trained model.'
    )
    parser.add_argument(
        '-face',
        action='store_true',
        help='Load the face identifier model with age and gender prediction.'
            'The model is supposed to be in "saved_models/Face.keras"'
             'Results will be saved under "predictions_face.png"'
    )
    parser.add_argument(
        '-facedetector',
        action='store_true',
        help='Load the face detector model with face detection only.'
             'The model is supposed to be in "saved_models/FaceDetector.keras"'
             'Results will be saved under "predictions_face_detector.png"'
    )
    parser.add_argument(
        '-noshow',
        action='store_true',
        help='Deactivates immediate printing of the result to the console.'
    )
    parser.add_argument(
        '-datafolder',
        type=str,
        action='store',
        help='If set, will use the images in the specified folder for the predictions.'
    )
    parser.print_help()
    args = parser.parse_args()

    # Directories
    face_directory = 'data/utk-face'
    non_face_directory = 'data/nonface/openimages'

    # Load random images
    images, true_labels = load_random_images(face_directory, non_face_directory)

    if args.face:
        # Load face model and show its predictions
        model = saving.load_model('saved_models/Face.keras')
        if not args.datafolder:
            show_random_predictions_face(images, true_labels, noshow = args.noshow)
        else:
            predict_all_images(load_all_images(args.datafolder), model, noshow=args.noshow)
    elif args.facedetector:
        # Load face detector model and show its predictions
        model = saving.load_model('saved_models/FaceDetector.keras')
        if not args.datafolder:
            show_random_predictions_face_detector(images, true_labels, noshow = args.noshow)
        else:
            predict_all_images(load_all_images(args.datafolder), model, noshow=args.noshow)


