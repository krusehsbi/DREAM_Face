from utils import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from models import MultitaskResNet

image_directory = 'data/utk-face/UTKFace'  # Update with your dataset path
images, labels = load_data(image_directory)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Separate the labels for each task (face/no face, age, gender)
y_train_face, y_train_age, y_train_gender = y_train[:, 0], y_train[:, 1], y_train[:, 2]
y_val_face, y_val_age, y_val_gender = y_val[:, 0], y_val[:, 1], y_val[:, 2]

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, {'face_output': y_train_face, 
                                                              'age_output': y_train_age, 
                                                              'gender_output': y_train_gender})).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, {'face_output': y_val_face, 
                                                          'age_output': y_val_age, 
                                                          'gender_output': y_val_gender})).batch(32)

# Create Model
model = MultitaskResNet(input_shape=(128, 128, 3))
model.build_model()
model.compile_model()
model.model.summary()

# Training and Evaluation
history = model.train(train_dataset, val_dataset, epochs=10)
results = model.evaluate(val_dataset)
model.save_model('saved_models/multitask_resnet_model.h5')