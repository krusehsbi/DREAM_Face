import numpy as np
from keras import layers, applications, metrics, losses, optimizers, callbacks, saving, ops, random
from utils import load_data, shuffle_arrays, DataGeneratorIdentifier, DataGeneratorDetector, PlotHistory
import matplotlib.pyplot as plt
from sklearn import model_selection
import keras

"""
This script implements a deep learning pipeline for face detection using Keras and EfficientNetB0.
The pipeline includes data preprocessing, model creation, training, evaluation, and inference.
"""

# Custom Keras layer to randomly convert an image to grayscale during training
@keras.saving.register_keras_serializable()
class RandomGrayscale(layers.Layer):
    def __init__(self, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability

    def call(self, inputs, training=True):
        """
        In training mode, randomly convert input images to grayscale with a given probability.
        This helps the model generalize better.
        """
        if training:
            if np.random.random() > self.probability:
                # Convert the image to grayscale using the luminance formula
                grayscale = ops.dot(inputs[..., :3], [0.2989, 0.5870, 0.1140])  # Grayscale formula
                grayscale = ops.expand_dims(grayscale, axis=-1)
                # Concatenate the grayscale values back to 3 channels (RGB) to maintain dimensions
                inputs = ops.concatenate((grayscale, grayscale, grayscale), axis=-1)
        return inputs

    def get_config(self):
        """
        Serialize the configuration of the RandomGrayscale layer so it can be saved.
        """
        config = super().get_config()
        config.update({"probability": self.probability})
        return config

# Preprocessing pipeline that applies random data augmentations
def preprocessing_pipeline(inputs):
    """
    Applies a sequence of random augmentations to the input images:
    - Random Zoom
    - Random Rotation
    - Random Horizontal Flip
    - Random Brightness
    - Random Contrast
    - Random Grayscale conversion
    """
    x = layers.RandomZoom(0.2)(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomBrightness(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    x = RandomGrayscale(probability=0.5)(x)
    return x

# Face detection model using EfficientNetB0 as a feature extractor
def FaceDetector(input_shape):
    """
    Creates and compiles a face detection model using EfficientNetB0 as a base model.
    The base model is pre-trained on ImageNet and frozen during training.
    """
    inputs = layers.Input(shape=input_shape)  # Input layer for the model
    x = preprocessing_pipeline(inputs)  # Apply the preprocessing pipeline

    # Load the EfficientNetB0 model (without top layers) for feature extraction
    basemodel = applications.EfficientNetB0(weights='imagenet', include_top=False)
    basemodel.trainable = False  # Freeze the base model layers during training
    x = basemodel(x)

    # Add custom layers on top of the base model
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)  # Batch normalization for better training stability
    x = layers.ReLU()(x)  # Activation function

    x = layers.Dropout(0.25)(x)  # Dropout to prevent overfitting
    outputs = layers.Dense(1, activation=None, name='face_output')(x)  # Output layer for binary classification

    model = keras.Model(inputs=inputs, outputs=outputs)  # Create the Keras model

    # Compile the model using Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=[metrics.BinaryAccuracy(name='face_accuracy')]  # Track binary accuracy during training
    )

    model.summary()  # Print the model summary to inspect the architecture
    return model

# Inference function for a batch of images
def infer_images(images, model=None):
    """
    Given a list of images, it makes predictions about whether each image contains a face or not.
    """
    if model is None:
        model = saving.load_model("saved_models/FaceDetector.keras")  # Load a pre-trained model

    for image in images:
        infer_image(image, model)  # Infer each image individually

# Inference function for a single image
def infer_image(image, model):
    """
    Makes a prediction for a single image and prints the result.
    """
    predictions = model.predict(ops.expand_dims(image, 0))  # Predict using the model
    score = float(ops.sigmoid(predictions[0][0]))  # Apply sigmoid to the logits for the binary output

    # Print the certainty score of whether the image contains a face
    print(f"This image contains a face with {100 * score:.2f}% certainty.")

    plt.imshow(image)  # Display the image
    plt.show()

# Main script
if __name__ == '__main__':
    # Define directories for face and non-face images
    face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
    non_face_directory = ['data/nonface']

    # Load and preprocess images and labels from the directories
    images, labels = load_data(face_directory, non_face_directory, deserialize_data=True,
                               serialize_data=True, preprocess_fnc=applications.efficientnet.preprocess_input)
    images, labels = shuffle_arrays(images, labels)  # Shuffle the images and labels for random training splits

    # Split data into training, validation, and test sets
    images_train, images_temp, labels_train, labels_temp = model_selection.train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)

    # Split the remaining 20% of the data into validation and test sets
    images_val, images_test, labels_val, labels_test = model_selection.train_test_split(images_temp, labels_temp, test_size=0.5,
                                                                        random_state=42)

    # Extract the labels for the face/no-face task
    labels_train_face = labels_train[:, 0]
    labels_val_face = labels_val[:, 0]
    labels_test_face = labels_test[:, 0]

    batch_size = 32  # Define the batch size for training

    # Create data generators for training, validation, and testing
    training_generator = DataGeneratorDetector(
        images_train,
        labels_train_face,
        batch_size)

    val_generator = DataGeneratorDetector(
        images_val,
        labels_val_face,
        batch_size)

    test_generator = DataGeneratorDetector(
        images_test,
        labels_test_face,
        batch_size)

    # Define the file path for model checkpointing
    checkpoint_filepath = '/tmp/checkpoints/checkpoint.faceDetector.keras'

    # Initialize the face detection model
    model = FaceDetector((128, 128, 3))

    # List of callbacks to use during training (for model checkpointing and early stopping)
    model_callbacks = []
    model_callbacks.append(callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_face_accuracy',  # Monitor validation accuracy for saving the best model
        mode='max',
        save_best_only=True  # Save only the best model based on validation accuracy
    ))

    model_callbacks.append(callbacks.EarlyStopping(
        monitor='val_face_accuracy',
        min_delta=0.001,  # Minimum change to qualify as an improvement
        patience=3,  # Stop training if no improvement for 3 epochs
        restore_best_weights=True,  # Restore weights from the best epoch
        mode="max"
    ))

    # Train the model
    history = model.fit(x=training_generator,
                        validation_data=val_generator,
                        epochs=500,  # Train for up to 500 epochs
                        callbacks=model_callbacks)

    # Evaluate the trained model on the test set
    result = model.evaluate(x=test_generator)
    print(result)

    # Save the trained model to disk
    model.save("saved_models/FaceDetector.keras")

    # Plot the training and validation loss over epochs
    PlotHistory(history)

    # Load the saved model and perform inference on random test images
    model = saving.load_model("saved_models/FaceDetector.keras")
    infer_images(images[np.random.choice(images.shape[0], 8, replace=False)], model)
