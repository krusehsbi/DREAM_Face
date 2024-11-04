from utils import load_data, deserialize_saved_data, serialize_loaded_data
from sklearn.model_selection import train_test_split
from models import MultitaskResNet
from keras import callbacks
import csv

FORCE_LOAD_DATA = True
SERIALIZE_DATA = False

# Define directories
face_directory = 'data/utk-face/UTKFace'
non_face_directory = 'data/nonface/imagenet_images'

# Load images and labels from both face and non-face directories
images, labels = None, None

if not FORCE_LOAD_DATA:
    images, labels = deserialize_saved_data()

if FORCE_LOAD_DATA or images is None or labels is None or len(images) == 0 or len(labels) == 0:
    images, labels = load_data(face_directory, non_face_directory)

    if SERIALIZE_DATA:
        serialize_loaded_data(images, labels)


# Step 1: Split data into training (80%) and test+validation (20%) sets
images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 2: Split the remaining 20% data into validation (10%) and test (10%) sets
images_val, images_test, labels_val, labels_test = train_test_split( images_temp, labels_temp, test_size=0.5, random_state=42)

# Separate the labels for each task (face/no face, age, gender)
labels_train_face, labels_train_age, labels_train_gender = labels_train[:, 0], labels_train[:, 1], labels_train[:, 2]
labels_val_face, labels_val_age, labels_val_gender = labels_val[:, 0], labels_val[:, 1], labels_val[:, 2]
labels_test_face, labels_test_age, labels_test_gender = labels_test[:, 0], labels_test[:, 1], labels_test[:, 2]

# Create Model
input_shape = (128, 128, 3)
model = MultitaskResNet(input_shape)
model.build(input_shape)
model.summary()

model.compile()



# Define early stopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=3,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
)

# Training and Evaluation
history = model.fit(x=images_train,
                    y=[labels_train_face, labels_train_age, labels_train_gender],
                    validation_data=(images_val, [labels_val_face, labels_val_age, labels_val_gender]),
                    epochs=500,
                    batch_size=64,
                    callbacks=[early_stopping])

results = model.evaluate(x=images_train,
                         y=[labels_train_face, labels_train_age, labels_train_gender],
                         batch_size=64,
                         return_dict=True)
print("Test results: ", results)

model.save('saved_models/multitask_resnet_model_dropout05.keras')

with open('saved_models/training_history_dropout05.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(history.history.keys())
    # Write data
    writer.writerows(zip(*history.history.values()))
