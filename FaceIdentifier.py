import numpy as np
from keras import models, layers, applications, metrics, losses, optimizers, callbacks, saving, ops
from utils import load_data, shuffle_arrays
import matplotlib.pyplot as plt
from sklearn import model_selection
from FaceDetector import preprocessing_pipeline
import csv


def MultitaskResNet(input_shape=(128, 128, 3), dropout_rate=0.25):
    inputs = layers.Input(shape=input_shape)
    x = preprocessing_pipeline(inputs)

    basemodel = applications.ResNet50(weights='imagenet', include_top=False)
    basemodel.trainable = False
    x = basemodel(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    face_output = layers.Dropout(rate=dropout_rate, name='face_dropout')(x)
    face_output = layers.Dense(1, activation=None, name='face_output')(face_output)

    age_output = layers.Dense(256, activation='relu', name='age_1')(x)
    age_output = layers.Dense(128, activation='relu', name='age_2')(age_output)
    age_output = layers.BatchNormalization(name='age_normalization')(age_output)
    age_output = layers.Dropout(rate=dropout_rate, name='age_dropout')(age_output)
    age_output = layers.Dense(1, activation='linear', name='age_output')(age_output)

    gender_output = layers.Dense(256, activation='relu', name='gender_1')(x)
    gender_output = layers.Dense(128, activation='relu', name='gender_2')(gender_output)
    gender_output = layers.BatchNormalization(name='gender_normalization')(gender_output)
    gender_output = layers.Dropout(rate=dropout_rate, name='gender_dropout')(gender_output)
    gender_output = layers.Dense(3, activation=None, name='gender_output')(gender_output)

    model = models.Model(inputs=inputs, outputs={'face_output' : face_output,
                                                 'age_output' : age_output,
                                                 'gender_output' : gender_output})

    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss={
            'face_output': losses.BinaryCrossentropy(from_logits=True),
            'age_output': losses.MeanAbsoluteError(),
            'gender_output': losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        metrics={
            'face_output': metrics.BinaryAccuracy(name='accuracy'),
            'age_output': metrics.MeanAbsoluteError(name='absolute_error'),
            'gender_output': metrics.SparseCategoricalAccuracy(name='accuracy'),
        })

    return model

def infer_images(images, model=None):
    if model is None:
        model = saving.load_model("saved_models/FaceDetector.keras")

    for image in images:
        infer_image(image, model)

def infer_image(image, model):
    predictions = model.predict(ops.expand_dims(image, 0))
    score_face = float(ops.sigmoid(predictions['face_output'][0]))
    score_age = float(ops.mean(predictions['age_output'][0]))
    score_gender = float(ops.argmax(predictions['gender_output'][0]))

    print(f"This image contains a face with {100 * score_face:.2f}% certainty.")

    if score_face > 0.5:
        print(f"The person has gender {"male" if score_gender == 0 else "female" } and is {score_age} years old.")


    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    # Define directories
    face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
    non_face_directory = ['data/nonface']

    # Load images and labels from both face and non-face directories
    images, labels = load_data(face_directory, non_face_directory, deserialize_data=True,
                               serialize_data=True, preprocess_fnc=None)
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

    model = saving.load_model("saved_models/Face.keras")
    infer_images(images[np.random.choice(images.shape[0], 8, replace=False)], model)

    checkpoint_filepath = '/tmp/checkpoints/checkpoint.face.keras'

    model = MultitaskResNet((128, 128, 3), 0.25)
    model.summary()
    # if os.path.exists(checkpoint_filepath):
    # model.load_weights(checkpoint_filepath)

    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='max',
        save_best_only=True
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5,
        restore_best_weights=True,
        mode="max"
    )

    history = model.fit(x=images_train,
                        y={'face_output': labels_train_face,
                           'age_output': labels_train_age,
                           'gender_output': labels_train_gender},
                        validation_data=(images_val,
                                         {'face_output': labels_val_face,
                                          'age_output': labels_val_age,
                                          'gender_output': labels_val_gender}),
                        epochs=5,
                        batch_size=128,
                        callbacks=[
                            checkpoint,
                            early_stopping
                        ])
    print(history)

    result = model.evaluate(
        x=images_test,
        y={'face_output': labels_val_face,
           'age_output': labels_val_age,
           'gender_output': labels_val_gender})

    print(result)

    model.save("saved_models/Face.keras")

    # Final evaluation on test set
    results = model.evaluate(x=images_test,
                             y={'face_output': labels_test_face,
                                'age_output': labels_test_age,
                                'gender_output': labels_test_gender},
                             batch_size=256,
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

    plt.plot(history.history['age_output_absolute_error'])
    plt.plot(history.history['val_age_output_absolute_error'])
    plt.title('Age Mean Absolute Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['age_output_loss'])
    plt.plot(history.history['val_age_output_loss'])
    plt.title('Age Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['face_output_accuracy'])
    plt.plot(history.history['val_face_output_accuracy'])
    plt.title('Face Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['face_output_loss'])
    plt.plot(history.history['val_face_output_loss'])
    plt.title('Face Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['gender_output_accuracy'])
    plt.plot(history.history['val_gender_output_accuracy'])
    plt.title('Gender Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['gender_output_loss'])
    plt.plot(history.history['val_gender_output_loss'])
    plt.title('Gender Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Total Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
