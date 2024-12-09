import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import wandb
from keras import models, layers, applications, metrics, losses, optimizers, callbacks, saving, ops, utils
from utils import load_data, shuffle_arrays, DataGeneratorIdentifier, PlotHistory
import matplotlib.pyplot as plt
from sklearn import model_selection
from FaceDetector import preprocessing_pipeline
import csv
import keras
from wandb.integration.keras import WandbMetricsLogger
import random

import tensorflow as tf

"""
Multi-Task Deep Learning Model for Face Analysis

This module implements a deep learning model that performs three simultaneous tasks:
1. Face Detection: Binary classification to detect presence of faces
2. Age Estimation: Regression to predict age
3. Gender Classification: Binary classification for gender prediction

The model architecture uses transfer learning with a pre-trained backbone (e.g., EfficientNet, ResNet)
and task-specific output branches. Each branch is optimized with custom loss functions and metrics
that handle invalid or missing data.

Key Components:
- Multiple backbone options (EfficientNet, ResNet, Inception, MobileNet)
- Custom loss functions with invalid data masking
- Data augmentation pipeline
- Training with early stopping and learning rate scheduling
- Weights & Biases integration for experiment tracking

Example usage:
    hyperparameters = SimpleNamespace(
        epochs=20,
        batch_size=32,
        dim=(128, 128),
        learning_rate=3e-4,
        dropout_rate=0.25,
        model='efficientnet-b0'
    )
    model = FaceIdentifier(
        input_shape=(256, 256, 3),
        dropout_rate=hyperparameters.dropout_rate,
        learning_rate=hyperparameters.learning_rate,
        model=hyperparameters.model
    )
"""


@keras.saving.register_keras_serializable()
def age_loss_fn(y_true, y_pred):
    """
    Custom loss function for age prediction that handles invalid age values.

    The function masks out ages >= 200 (considered invalid) and computes MSE
    only on valid entries. This prevents invalid ages from affecting model training.

    Args:
        y_true (tensor): Ground truth age values
        y_pred (tensor): Predicted age values

    Returns:
        float: Mean squared error computed only on valid age entries
    """
    y_pred = y_pred * ops.cast(ops.less(y_true, 200), y_pred.dtype)  # Mask invalid age values (>= 200)
    y_true = y_true * ops.cast(ops.less(y_true, 200), y_true.dtype)
    return losses.mean_squared_error(y_true, y_pred)


@keras.saving.register_keras_serializable()
def age_metric(y_true, y_pred):
    """
    Custom metric for age prediction that handles invalid age values.

    Computes Mean Absolute Error (MAE) only on valid age entries (< 200).
    Used during training to monitor model performance on the age prediction task.

    Args:
        y_true (tensor): Ground truth age values
        y_pred (tensor): Predicted age values

    Returns:
        float: Mean absolute error computed only on valid age entries
    """
    mask = ops.less(y_true, 200)
    mask_pred = ops.expand_dims(mask, axis=-1)

    y_pred = ops.where(mask_pred, y_pred, ops.zeros_like(y_pred))
    y_true = ops.where(mask, y_true, ops.zeros_like(y_true))
    return metrics.mean_absolute_error(y_true, y_pred)


@keras.saving.register_keras_serializable()
def gender_loss_fn(y_true, y_pred):
    """
    Custom loss function for gender prediction that handles invalid labels.

    Computes binary cross-entropy only on valid gender labels (< 2).
    Gender values >= 2 are considered invalid and masked out.

    Args:
        y_true (tensor): Ground truth gender labels
        y_pred (tensor): Predicted gender probabilities

    Returns:
        float: Binary cross-entropy computed only on valid gender entries
    """
    y_pred = y_pred * ops.cast(ops.less(y_true, 2), y_pred.dtype)
    y_true = y_true * ops.cast(ops.less(y_true, 2), y_true.dtype)
    return losses.binary_crossentropy(y_true, y_pred)


@keras.saving.register_keras_serializable()
def gender_metric(y_true, y_pred):
    """
    Custom metric for gender prediction that handles invalid labels.

    Computes binary accuracy only on valid gender labels (< 2).
    Used during training to monitor model performance on gender classification.

    Args:
        y_true (tensor): Ground truth gender labels
        y_pred (tensor): Predicted gender probabilities

    Returns:
        float: Binary accuracy computed only on valid gender entries
    """
    mask = ops.less(y_true, 2)
    mask_pred = ops.expand_dims(mask, axis=-1)

    y_pred = ops.where(mask_pred, y_pred, ops.zeros_like(y_pred))
    y_true = ops.where(mask, y_true, ops.zeros_like(y_true))
    return metrics.binary_accuracy(y_true, y_pred)


def FaceIdentifier(
        input_shape=(256, 256, 3),
        dropout_rate=0.25,
        learning_rate=3e-4,
        model='efficientnet-b0'
):
    """
    Creates and compiles a multi-task model for face analysis.

    The model uses a pre-trained backbone followed by task-specific branches for
    face detection, age estimation, and gender classification. The backbone's weights
    are frozen for transfer learning.

    Architecture:
    - Backbone: Pre-trained model (EfficientNet/ResNet/Inception/MobileNet)
    - Face Branch: Global pooling -> Dropout -> Dense(1, sigmoid)
    - Age Branch: Dense(2024) -> Dense(1024) -> Dense(1, relu)
    - Gender Branch: Dense(2024) -> Dense(1024) -> Dense(1, sigmoid)

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Initial learning rate for Adam optimizer
        model (str): Backbone architecture name ('efficientnet-b0', 'resnet50', etc.)

    Returns:
        keras.Model: Compiled multi-task model ready for training

    Raises:
        Exception: If specified backbone model is not supported
    """
    inputs = layers.Input(shape=input_shape)

    # Initialize backbone model and preprocessing function
    basemodel = None
    preprocessing = None
    match model:
        case 'efficientnet-b0':
            basemodel = applications.efficientnet.EfficientNetB0(weights='imagenet', include_top=False)
            preprocessing = applications.efficientnet.preprocess_input
        case 'efficientnet-b4':
            basemodel = applications.efficientnet.EfficientNetB4(weights='imagenet', include_top=False)
            preprocessing = applications.efficientnet.preprocess_input
        case 'resnet50':
            basemodel = applications.ResNet50(weights='imagenet', include_top=False)
            preprocessing = applications.resnet.preprocess_input
        case 'resnet101':
            basemodel = applications.ResNet101(weights='imagenet', include_top=False)
            preprocessing = applications.resnet.preprocess_input
        case 'inception':
            basemodel = applications.InceptionV3(weights='imagenet', include_top=False)
            preprocessing = applications.inception_v3.preprocess_input
        case 'mobilenet':
            basemodel = applications.MobileNetV2(weights='imagenet', include_top=False)
            preprocessing = applications.mobilenet_v2.preprocess_input

    if basemodel is None:
        raise Exception('Unsupported base model specified.')

    basemodel.trainable = False  # Freeze backbone weights

    # Apply preprocessing and backbone
    x = preprocessing_pipeline(inputs, preprocessing)
    x = basemodel(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Define face output branch (binary classification)
    face_output = layers.Dropout(rate=dropout_rate, name='face_dropout')(x)
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(face_output)

    # Define age output branch (regression)
    age_output = layers.Dense(256, activation='relu', name='age_1')(x)
    age_output = layers.Dropout(rate=dropout_rate, name='age_dropout')(age_output)
    age_output = layers.Dense(1, activation='relu', name='age_output')(age_output)

    # Define gender output branch (multi-class classification)
    gender_output = layers.Dense(256, activation='relu', name='gender_1')(x)
    gender_output = layers.Dense(128, activation='relu', name='gender_2')(gender_output)
    gender_output = layers.Dropout(rate=dropout_rate, name='gender_dropout')(gender_output)
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_output)

    # Create and compile model
    model = models.Model(
        inputs=inputs,
        outputs={
            'face_output': face_output,
            'age_output': age_output,
            'gender_output': gender_output
        }
    )

    model.compile(
        run_eagerly=True,
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss={
            'face_output': losses.BinaryCrossentropy(),
            'age_output': age_loss_fn,
            'gender_output': gender_loss_fn,
        },
        metrics={
            'face_output': metrics.BinaryAccuracy(),
            'age_output': age_metric,
            'gender_output': gender_metric,
        }
    )

    utils.plot_model(model)
    return model


def infer_images(images, model, show=True):
    """
    Performs batch inference on multiple images.

    Args:
        images (list): List of preprocessed input images
        model (keras.Model): Trained model for inference
        show (bool): Whether to display images during inference

    Returns:
        list: List of prediction results for each image
    """
    results = []
    for image in images:
        results.append(infer_image(image, model, show))
    return results


def infer_image(image, model, show=True):
    """
    Performs inference on a single image.

    Makes predictions for face detection, age, and gender.
    Displays the image and prints prediction results if show=True.

    Args:
        image (numpy.ndarray): Preprocessed input image
        model (keras.Model): Trained model for inference
        show (bool): Whether to display the image and print results

    Returns:
        str: Formatted string containing prediction results
    """
    if show:
        plt.imshow(image)
        plt.show()

    predictions = model.predict(ops.expand_dims(image, 0))
    score_face = float(predictions['face_output'][0][0])
    score_age = round(predictions['age_output'][0][0])
    score_gender = float(predictions['gender_output'][0][0])

    label = f"This image contains a face with {100 * score_face:.2f}% certainty."
    print(label)

    if score_face > 0.5:
        additional_label = f"The person has gender {'male' if score_gender <= 0.5 else 'female'} and is {score_age} years old."
        print(additional_label)
        label += '\n' + additional_label

    return label


def train_and_evaluate_hyperparameters(hyperparameters, x, y, model_save_path):
    """
    Trains and evaluates the model with given hyperparameters.

    Performs the following steps:
    1. Splits data into train/validation/test sets
    2. Creates data generators for each set
    3. Trains the model with early stopping and LR scheduling
    4. Saves model checkpoints and training history
    5. Performs inference on random test samples

    Args:
        hyperparameters (SimpleNamespace): Training hyperparameters
        x (numpy.ndarray): Input images
        y (numpy.ndarray): Target labels
        model_save_path (Path): Directory to save model and results
    """
    # Split data into train/val/test sets
    images_train, images_temp, labels_train, labels_temp = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=random.randint(0, 20000)
    )

    images_val, images_test, labels_val, labels_test = model_selection.train_test_split(
        images_temp, labels_temp, test_size=0.5, random_state=random.randint(0, 20000)
    )

    # Separate labels for each task
    labels_train_face, labels_train_age, labels_train_gender = (
        labels_train[:, 0], labels_train[:, 1], labels_train[:, 2]
    )
    labels_val_face, labels_val_age, labels_val_gender = (
        labels_val[:, 0], labels_val[:, 1], labels_val[:, 2]
    )
    labels_test_face, labels_test_age, labels_test_gender = (
        labels_test[:, 0], labels_test[:, 1], labels_test[:, 2]
    )

    # Load existing model if available
    if os.path.exists("saved_models/Face.keras"):
        try:
            model = saving.load_model("saved_models/Face.keras")
            infer_images(images[np.random.choice(images.shape[0], 8, replace=False)], model)
        except Exception as e:
            print(e)

    # Create data generators
    training_generator = DataGeneratorIdentifier(
        images_train,
        labels_train_face,
        labels_train_age,
        labels_train_gender,
        hyperparameters.batch_size
    )

    val_generator = DataGeneratorIdentifier(
        images_val,
        labels_val_face,
        labels_val_age,
        labels_val_gender,
        hyperparameters.batch_size
    )

    test_generator = DataGeneratorIdentifier(
        images_test,
        labels_test_face,
        labels_test_age,
        labels_test_gender,
        hyperparameters.batch_size
    )

    # Initialize model
    checkpoint_filepath = '/tmp/checkpoints/checkpoint.face.keras'
    model = FaceIdentifier(
        input_shape=(*hyperparameters.dim, 3),
        dropout_rate=hyperparameters.dropout_rate,
        learning_rate=hyperparameters.learning_rate,
        model=hyperparameters.model,
    )
    model.summary()

    def scheduler(epoch, lr):
        return float(lr * hyperparameters.learning_rate_factor)

    # Define callbacks for training
    model_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=5,
            restore_best_weights=True,
            mode="min"
        ),
        callbacks.LearningRateScheduler(scheduler)
    ]

    # Initialize Weights & Biases tracking if available
    try:
        wandb.init(
            project="DREAM_Face",
            config={
                "epochs": hyperparameters.epochs,
                "batch_size": hyperparameters.batch_size,
                "start_learning_rate": hyperparameters.learning_rate,
                "learning_rate_factor": hyperparameters.learning_rate_factor,
                "dropout": hyperparameters.dropout_rate,
                "base_model": hyperparameters.model,
            })
        model_callbacks.append(WandbMetricsLogger())
    except Exception as e:
        print("No wandb callback added.")

    # Train the model
    history = model.fit(
        x=training_generator,
        validation_data=val_generator,
        epochs=hyperparameters.epochs,
        callbacks=model_callbacks
    )

    # Plot training history
    PlotHistory(history)

    # Evaluate model on test set
    result = model.evaluate(x=test_generator)

    # Save the trained model
    model.save(model_save_path / "Face.keras")

    # Load model and perform inference on random samples
    model = saving.load_model(model_save_path / "Face.keras")
    infer_images(x[np.random.choice(x.shape[0], 8, replace=False)], model)

    # Save training history to CSV
    with open(model_save_path / 'training_history_dropout_face.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(history.history.keys())
        # Write data
        writer.writerows(zip(*history.history.values()))

if __name__ == '__main__':
    """
    Main execution block for training and evaluating the face analysis model.

    Sets up hyperparameters, data paths, and runs the training pipeline.

    Hyperparameters:
    - epochs: Number of training epochs
    - batch_size: Number of samples per training batch
    - dim: Input image dimensions
    - learning_rate: Initial learning rate
    - dropout_rate: Dropout rate for regularization
    - learning_rate_factor: Factor for learning rate decay
    - model: Choice of backbone architecture
    """
    # Define hyperparameters
    hyperparameters = SimpleNamespace(
        epochs=20,
        batch_size=32,
        dim=(128, 128),
        learning_rate=3e-4,
        dropout_rate=0.25,
        learning_rate_factor=0.9,
        model='efficientnet-b0',  # Options: efficientnet-b0/b4, resnet50/101, inception, mobilenet
    )

    # Create save directory
    model_save_path = Path("saved_models")
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Define data directories
    face_directory = ['data/utk-face/UTKFace']
    non_face_directory = ['data/nonface']

    # Load and preprocess data
    images, labels = load_data(
        face_directory,
        non_face_directory,
        deserialize_data=True,
        serialize_data=True,
        dim=hyperparameters.dim
    )

    # Shuffle data
    images, labels = shuffle_arrays(images, labels)

    # Train and evaluate model
    train_and_evaluate_hyperparameters(hyperparameters, images, labels, model_save_path)
