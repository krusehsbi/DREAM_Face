import os
from pathlib import Path
import random
from random import shuffle

from utils import load_data, shuffle_arrays
from sklearn.model_selection import train_test_split
from models import MultitaskResNet, MultitaskResNetDropout
from keras import callbacks
import csv
import matplotlib.pyplot as plt
from utils import DataGenerator
import numpy as np
from face_inference import infer_image

DESERIALIZE_DATA = True
SERIALIZE_DATA = True

# Define directories
face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
non_face_directory = ['data/nonface']

# Load images and labels from both face and non-face directories
images, labels = load_data(face_directory, non_face_directory, deserialize_data=DESERIALIZE_DATA,
                           serialize_data=SERIALIZE_DATA)
images, labels = shuffle_arrays(images, labels)

# Step 1: Split data into training (80%) and test+validation (20%) sets
images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 2: Split the remaining 20% data into validation (10%) and test (10%) sets
images_val, images_test, labels_val, labels_test = train_test_split(images_temp, labels_temp, test_size=0.5,
                                                                    random_state=42)

# Separate the labels for each task (face/no face, age, gender)
labels_train_face, labels_train_age, labels_train_gender = labels_train[:, 0], labels_train[:, 1], labels_train[:, 2]
labels_val_face, labels_val_age, labels_val_gender = labels_val[:, 0], labels_val[:, 1], labels_val[:, 2]
labels_test_face, labels_test_age, labels_test_gender = labels_test[:, 0], labels_test[:, 1], labels_test[:, 2]

# Instantiate and compile the model
model = MultitaskResNetDropout(input_shape=(128, 128, 3))
model.build()
model.summary()
model.compile_default()

# Define early stopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
    # Whether to restore model weights from the epoch with the best value of the monitored quantity
)

# Initialize the generator
train_generator = DataGenerator(images_train, labels_train_face, labels_train_age, labels_train_gender, batch_size=128)
val_generator = DataGenerator(images_val, labels_val_face, labels_val_age, labels_val_gender, batch_size=128)

# Train the model using the generator
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping]
)

print(history.history.keys())

plt.plot(history.history['age_output_mae'])
plt.plot(history.history['val_age_output_mae'])
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

# Final evaluation on test set
results = model.evaluate(x=images_test,
                         y={'face_output': labels_test_face,
                            'age_output': labels_test_age,
                            'gender_output': labels_test_gender},
                         batch_size=256,
                         return_dict=True)
print("Test results:", results)

test_images = images[np.random.choice(images.shape[0], 8, replace=False)]
for image in test_images:
    infer_image(image, model)

# Save the trained model
(Path(__file__).parent/'saved_models').mkdir(exist_ok=True)
(Path(__file__).parent/'saved_weights').mkdir( exist_ok=True)
model.save('saved_models/multitask_resnet_model_dropout_face.keras')
model.save_weights('saved_weights/multitask_resnet_model_dropout_face.weights.h5')

with open('saved_models/training_history_dropout_face.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(history.history.keys())
    # Write data
    writer.writerows(zip(*history.history.values()))
