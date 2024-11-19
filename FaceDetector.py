import numpy as np
from keras import layers, applications, metrics, losses, optimizers, callbacks, saving, ops, random

from utils import load_data, shuffle_arrays, DataGeneratorIdentifier, DataGeneratorDetector, PlotHistory

import matplotlib.pyplot as plt

from sklearn import model_selection

import keras

@keras.saving.register_keras_serializable()
class RandomGrayscale(layers.Layer):
    def __init__(self, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability

    def call(self, inputs, training=True):
        # Randomly decide whether to convert to grayscale
        if training:
            if np.random.random() > self.probability:
                # Convert to grayscale and back to RGB to keep dimensions the same
                grayscale = ops.dot(inputs[..., :3], [0.2989, 0.5870, 0.1140])
                grayscale = ops.expand_dims(grayscale, axis=-1)
                inputs = ops.concatenate((grayscale, grayscale, grayscale), axis=-1)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"probability": self.probability})
        return config

def preprocessing_pipeline(inputs):
    x = layers.RandomZoom(0.2)(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomBrightness(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    x = RandomGrayscale(probability=0.5)(x)
    return x

def FaceDetector(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = preprocessing_pipeline(inputs)

    basemodel = applications.EfficientNetB0(weights='imagenet', include_top=False)
    basemodel.trainable = False
    x = basemodel(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation=None, name='face_output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=[metrics.BinaryAccuracy(name='face_accuracy')]
    )

    model.summary()

    return model

def infer_images(images, model=None):
    if model is None:
        model = saving.load_model("saved_models/FaceDetector.keras")

    for image in images:
        infer_image(image, model)

def infer_image(image, model):
    predictions = model.predict(ops.expand_dims(image, 0))
    score = float(ops.sigmoid(predictions[0][0]))

    print(f"This image contains a face with {100 * score:.2f}% certainty.")

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    # Define directories
    face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
    non_face_directory = ['data/nonface']

    # Load images and labels from both face and non-face directories
    images, labels = load_data(face_directory, non_face_directory, deserialize_data=True,
                               serialize_data=True, preprocess_fnc=applications.efficientnet.preprocess_input)
    images, labels = shuffle_arrays(images, labels)

    # Step 1: Split data into training (80%) and test+validation (20%) sets
    images_train, images_temp, labels_train, labels_temp = model_selection.train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)

    # Step 2: Split the remaining 20% data into validation (10%) and test (10%) sets
    images_val, images_test, labels_val, labels_test = model_selection.train_test_split(images_temp, labels_temp, test_size=0.5,
                                                                        random_state=42)


    # Separate the labels for each task (face/no face, age, gender)
    labels_train_face = labels_train[:, 0]
    labels_val_face = labels_val[:, 0]
    labels_test_face = labels_test[:, 0]

    batch_size = 32
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

    checkpoint_filepath = '/tmp/checkpoints/checkpoint.faceDetector.keras'

    model = FaceDetector((128, 128, 3))
    #if os.path.exists(checkpoint_filepath):
        #model.load_weights(checkpoint_filepath)

    model_callbacks = []
    model_callbacks.append(callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_face_accuracy',
        mode='max',
        save_best_only=True
    ))

    model_callbacks.append(callbacks.EarlyStopping(
        monitor='val_face_accuracy',
        min_delta=0.001,
        patience=3,
        restore_best_weights=True,
        mode="max"
    ))

    history = model.fit(x=training_generator,
              validation_data=val_generator,
              epochs=500,
              callbacks=model_callbacks)

    result = model.evaluate(x=test_generator)
    print(result)

    model.save("saved_models/FaceDetector.keras")

    PlotHistory(history)

    model = saving.load_model("saved_models/FaceDetector.keras")

    infer_images(images[np.random.choice(images.shape[0], 8, replace=False)], model)