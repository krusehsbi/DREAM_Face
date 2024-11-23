import os
import numpy as np
import wandb
from keras import models, layers, applications, metrics, losses, optimizers, callbacks, saving, ops, utils, backend, random
from utils import load_data, shuffle_arrays, DataGeneratorIdentifier, PlotHistory
import matplotlib.pyplot as plt
from sklearn import model_selection
from FaceDetector import preprocessing_pipeline
import csv
import keras
from wandb.integration.keras import WandbMetricsLogger

"""
This module contains a multi-task deep learning model for face detection, age estimation, and gender classification. 
The model uses a pre-trained EfficientNetB0 architecture as a base model, and is extended with custom output layers for each task.

Key Features:
- **Face Detection**: A binary classification task to detect the presence of a face in an image.
- **Age Estimation**: A regression task to predict the age of a person in the image.
- **Gender Classification**: A multi-class classification task to predict the gender of the person in the image.
"""

# Custom loss and metric functions for the age and gender tasks.
@keras.saving.register_keras_serializable()
def age_loss_fn(y_true, y_pred):
    """
    Custom loss function for age prediction. It computes Mean Squared Error only for valid age values (age < 200).

    Args:
    - y_true: Ground truth labels (age values).
    - y_pred: Predicted age values.

    Returns:
    - loss: Mean squared error between true and predicted ages for valid entries.
    """
    y_pred = y_pred * ops.cast(ops.less(y_true, 200), y_pred.dtype)  # Mask invalid age values (>= 200)
    y_true = y_true * ops.cast(ops.less(y_true, 200), y_true.dtype)
    return losses.mean_squared_error(y_true, y_pred)


@keras.saving.register_keras_serializable()
def age_metric(y_true, y_pred):
    """
    Custom metric function for age prediction. It computes Mean Absolute Error only for valid age values (age < 200).

    Args:
    - y_true: Ground truth labels (age values).
    - y_pred: Predicted age values.

    Returns:
    - metric: Mean absolute error between true and predicted ages for valid entries.
    """
    mask = ops.less(y_true, 200)
    mask_pred = ops.expand_dims(mask, axis=-1)

    y_pred = ops.where(mask_pred, y_pred, ops.zeros_like(y_pred))
    y_true = ops.where(mask, y_true, ops.zeros_like(y_true))
    return metrics.mean_absolute_error(y_true, y_pred)


@keras.saving.register_keras_serializable()
def gender_loss_fn(y_true, y_pred):
    """
    Custom loss function for gender prediction. It computes Binary Cross-Entropy only for valid gender labels.

    Args:
    - y_true: Ground truth labels (gender values).
    - y_pred: Predicted gender probabilities.

    Returns:
    - loss: Binary cross-entropy between true and predicted gender labels.
    """
    y_pred = y_pred * ops.cast(ops.less(y_true, 2), y_pred.dtype)  # Mask invalid gender values (>= 2)
    y_true = y_true * ops.cast(ops.less(y_true, 2), y_true.dtype)
    return losses.binary_crossentropy(y_true, y_pred)


@keras.saving.register_keras_serializable()
def gender_metric(y_true, y_pred):
    """
    Custom metric function for gender prediction. It computes Binary Accuracy only for valid gender labels.

    Args:
    - y_true: Ground truth labels (gender values).
    - y_pred: Predicted gender probabilities.

    Returns:
    - metric: Binary accuracy between true and predicted gender labels.
    """

    mask = ops.less(y_true, 2)
    mask_pred = ops.expand_dims(mask, axis=-1)

    y_pred = ops.where(mask_pred, y_pred, ops.zeros_like(y_pred))
    y_true = ops.where(mask, y_true, ops.zeros_like(y_true))
    return metrics.binary_accuracy(y_true, y_pred)

def FaceIdentifier(input_shape=(128, 128, 3), dropout_rate=0.25):
    """
    Defines and compiles a multi-task model for face detection, age estimation, and gender classification.
    
    Args:
    - input_shape: Shape of the input images (default: (128, 128, 3)).
    - dropout_rate: Dropout rate for regularization (default: 0.25).
    
    Returns:
    - model: Compiled Keras model for face identification, age, and gender prediction.
    """
    inputs = layers.Input(shape=input_shape)

    # Apply preprocessing pipeline (augmentations, etc.)
    x = preprocessing_pipeline(inputs)

    # Load pre-trained EfficientNetB0 as the base model (frozen weights)
    basemodel = applications.efficientnet.EfficientNetB0(weights='imagenet', include_top=False)
    basemodel.trainable = False
    x = basemodel(x)

    # Add global average pooling and batch normalization
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

    # Combine all branches into a final model
    model = models.Model(inputs=inputs, outputs={'face_output': face_output,
                                                 'age_output': age_output,
                                                 'gender_output': gender_output})

    # Compile the model with respective loss functions and metrics
    model.compile(
        run_eagerly=False,
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss={
            'face_output': losses.BinaryCrossentropy(),
            'age_output': age_loss_fn,
            'gender_output': gender_loss_fn,
        },
        metrics={
            'face_output': metrics.BinaryAccuracy(),
            'age_output': age_metric,
            'gender_output': gender_metric,
        })

    # Plot the model architecture for visualization
    utils.plot_model(model)

    return model

def infer_images(images, model, show=True):
    """
    Performs inference on a list of images using the given model.

    Args:
    - images: List of images for inference.
    - model: Pre-trained Keras model for inference.
    - show: Whether to display the image during inference (default: True).

    Returns:
    - None
    """
    results = []
    for image in images:
         results.append(infer_image(image, model, show))
    return results


def infer_image(image, model, show=True):
    """
    Performs inference on a single image using the given model and prints the result.

    Args:
    - image: Single image for inference.
    - model: Pre-trained Keras model for inference.
    - show: Whether to display the image during inference (default: True).

    Returns:
    - label: The result label with face, age, and gender prediction.
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


if __name__ == '__main__':
    # Define directories
    face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
    non_face_directory = ['data/nonface']

    # Load images and labels from both face and non-face directories
    images, labels = load_data(face_directory, non_face_directory, deserialize_data=True,
                               serialize_data=True, preprocess_fnc=applications.efficientnet.preprocess_input)
    images, labels = shuffle_arrays(images, labels)

    # Step 1: Split data into training (80%) and test+validation (20%) sets
    images_train, images_temp, labels_train, labels_temp = model_selection.train_test_split(images,
                                                                                            labels,
                                                                                            test_size=0.2,
                                                                                            random_state=42)

    # Step 2: Split the remaining 20% data into validation (10%) and test (10%) sets
    images_val, images_test, labels_val, labels_test = model_selection.train_test_split(images_temp,
                                                                                        labels_temp,
                                                                                        test_size=0.5,
                                                                                        random_state=42)

    # Separate the labels for each task (face/no face, age, gender)
    labels_train_face, labels_train_age, labels_train_gender = (
        labels_train[:, 0],
        labels_train[:, 1],
        labels_train[:, 2])
    labels_val_face, labels_val_age, labels_val_gender = labels_val[:, 0], labels_val[:, 1], labels_val[:, 2]
    labels_test_face, labels_test_age, labels_test_gender = labels_test[:, 0], labels_test[:, 1], labels_test[:, 2]

    if os.path.exists("saved_models/Face.keras"):
        try:
            model = saving.load_model("saved_models/Face.keras")
            infer_images(images[np.random.choice(images.shape[0], 8, replace=False)], model)
        except Exception as e:
            print(e)


    batch_size = 32
    preprocess = applications.efficientnet.preprocess_input
    training_generator = DataGeneratorIdentifier(
        images_train,
        labels_train_face,
        labels_train_age,
        labels_train_gender,
        batch_size)

    val_generator = DataGeneratorIdentifier(
        images_val,
        labels_val_face,
        labels_val_age,
        labels_val_gender,
        batch_size)

    test_generator = DataGeneratorIdentifier(
        images_test,
        labels_test_face,
        labels_test_age,
        labels_test_gender,
        batch_size)

    checkpoint_filepath = '/tmp/checkpoints/checkpoint.face.keras'

    model = FaceIdentifier((128, 128, 3), 0.25)
    model.summary()
    # if os.path.exists(checkpoint_filepath):
    # model.load_weights(checkpoint_filepath)

    model_callbacks = []

    model_callbacks.append(callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    ))

    model_callbacks.append(callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5,
        restore_best_weights=True,
        mode="min"
    ))


    def scheduler(epoch, lr):
        return float(lr * ops.exp(-0.1))

    model_callbacks.append(callbacks.LearningRateScheduler(scheduler))

    try:
        wandb.init(config={'bs': 12})
        model_callbacks.append(WandbMetricsLogger())
    except Exception as e:
        print("No wandb callback added.")

    history = model.fit(x=training_generator,
                        validation_data=val_generator,
                        epochs=500,
                        callbacks=model_callbacks)
    print(history)

    result = model.evaluate(x=test_generator)

    print(result)

    model.save("saved_models/Face.keras")

    # Final evaluation on test set
    results = model.evaluate(x=test_generator,
                             return_dict=True)
    print(results)

    model = saving.load_model("saved_models/Face.keras")
    infer_images(images[np.random.choice(images.shape[0], 8, replace=False)], model)

    with open('saved_models/training_history_dropout_face.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(history.history.keys())
        # Write data
        writer.writerows(zip(*history.history.values()))

    PlotHistory(history)
