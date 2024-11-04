from utils import load_data
from sklearn.model_selection import train_test_split
from models import MultitaskResNet
from keras import callbacks
import csv
import matplotlib.pyplot as plt

DESERIALIZE_DATA = True
SERIALIZE_DATA = True

# Define directories
face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
non_face_directory = ['data/nonface/imagenet_images']

# Load images and labels from both face and non-face directories
images, labels = load_data(face_directory, non_face_directory, deserialize_data=DESERIALIZE_DATA,
                           serialize_data=SERIALIZE_DATA)
# image_alt, labels_alt = utils_alt.load_data(face_directory[0], non_face_directory[0])

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
model = MultitaskResNet(input_shape=(128, 128, 3))
model.build()
model.summary()
model.compile()

# Define early stopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
    # Whether to restore model weights from the epoch with the best value of the monitored quantity
)

# Train the model
history = model.fit(x=images_train,
                      y={'face_output': labels_train_face,
                         'age_output': labels_train_age,
                         'gender_output': labels_train_gender},
                      validation_data=(
                          images_val,
                          {'face_output': labels_val_face,
                           'age_output': labels_val_age,
                           'gender_output': labels_val_gender}),
                      epochs=500,
                      batch_size=64,
                      callbacks=[early_stopping])

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
                         batch_size=64,
                         return_dict=True)
print("Test results:", results)

# Save the trained model
model.save('saved_models/multitask_resnet_model_dropout0502.keras')

with open('saved_models/training_history_dropout0502.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(history.history.keys())
    # Write data
    writer.writerows(zip(*history.history.values()))
