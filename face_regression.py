from utils import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from models import MultitaskResNet, MultitaskResNetBaseLine
import csv
from utils import data_augmentation

# Define directories
face_directory = 'data/utk-face/UTKFace'
non_face_directory = 'data/nonface/imagenet_images'

# Load images and labels from both face and non-face directories
images, labels = load_data(face_directory, non_face_directory)

# Step 1: Split data into training (80%) and test+validation (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 2: Split the remaining 20% data into validation (10%) and test (10%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# This achieves the final distribution: 80% train, 10% validation, 10% test

# Separate labels for each task (face/no face, age, gender) for each set
y_train_face, y_train_age, y_train_gender = y_train[:, 0], y_train[:, 1], y_train[:, 2]
y_val_face, y_val_age, y_val_gender = y_val[:, 0], y_val[:, 1], y_val[:, 2]
y_test_face, y_test_age, y_test_gender = y_test[:, 0], y_test[:, 1], y_test[:, 2]

# Create TensorFlow datasets for train, validation, and test sets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, {'face_output': y_train_face, 
                                                              'age_output': y_train_age, 
                                                              'gender_output': y_train_gender}))

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
train_dataset = train_dataset.batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, {'face_output': y_val_face, 
                                                          'age_output': y_val_age, 
                                                          'gender_output': y_val_gender})).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, {'face_output': y_test_face, 
                                                            'age_output': y_test_age, 
                                                            'gender_output': y_test_gender})).batch(64)

# Instantiate and compile the model
model = MultitaskResNetBaseLine(input_shape=(128, 128, 3))
model.build_model()
model.compile_model()

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=3,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
)

# Train the model
history = model.train(train_dataset, val_dataset, epochs=500, callbacks=[early_stopping])

# Final evaluation on test set
test_results = model.evaluate(test_dataset)
print("Test results:", test_results)

# Save the trained model
model.save_model('saved_models/multitask_resnet_model_baseline.h5')

with open('saved_models/training_baseline.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(history.history.keys())
    # Write data
    writer.writerows(zip(*history.history.values()))