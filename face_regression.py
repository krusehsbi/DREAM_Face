from utils import load_data
from sklearn.model_selection import train_test_split
from models import MultitaskResNet

image_directory = 'data/utk-face'  # Update with your dataset path
images, labels = load_data(image_directory)

# Split the dataset into training and validation sets
images_train, images_val, labels_train, labels_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Separate the labels for each task (face/no face, age, gender)
labels_train_face, labels_train_age, labels_train_gender = labels_train[:, 0], labels_train[:, 1], labels_train[:, 2]
labels_val_face, labels_val_age, labels_val_gender = labels_val[:, 0], labels_val[:, 1], labels_val[:, 2]

# Create Model
model = MultitaskResNet(input_shape=(128, 128, 3))
model.compile()
model.summary()

# Training and Evaluation
history = model.fit(x=images_train,
                    y=[labels_train_face, labels_train_age, labels_train_gender],
                    validation_data=(images_val, [labels_val_face, labels_val_age, labels_val_gender]),
                    epochs=10,
                    batch_size=32)
results = model.evaluate(x=images_train,
                         y=[labels_train_face, labels_train_age, labels_train_gender],
                         batch_size=32,
                         return_dict=True)

model.save('saved_models/multitask_resnet_model.h5')
