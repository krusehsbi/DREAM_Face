import matplotlib.pyplot as plt

from models import MultitaskResNetDropout
from utils import *
import numpy as np
from keras import saving, ops, applications
from tkinter import *


DESERIALIZE_DATA = True
SERIALIZE_DATA = True

model_weights_to_load = 'multitask_resnet_model_dropout_face.weights.h5'
print(f"Loading model weights {model_weights_to_load}")
model = MultitaskResNetDropout(input_shape=(128, 128, 3))
model.load_weights(f"saved_weights/{model_weights_to_load}")
model.compile_default()

# Define directories
face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
non_face_directory = ['data/nonface']

# Load images and labels from both face and non-face directories
images, labels = load_data(face_directory, non_face_directory, deserialize_data=DESERIALIZE_DATA,
                           serialize_data=SERIALIZE_DATA)
images, labels = shuffle_arrays(images, labels)

test_images = images[np.random.choice(images.shape[0], 8, replace=False)]

for image in test_images:
    plt.imshow(image)
    plt.show()
    predictions = model.predict(applications.resnet.preprocess_input(ops.expand_dims(image, 0)))
    print(predictions)
    face_likeliness = float(ops.sigmoid(predictions['face_output'][0]))
    if face_likeliness > 0.5:
        age = float(ops.relu(predictions['age_output']))
        gender = float(ops.argmax(predictions['gender_output']))
        print(f"The image contains a face with {100 * face_likeliness:.2f}% confidence."
              f"The person is {age} years old."
              f"It has the gender {float(gender)}")
    else:
        print(f"The image contains no face with {100 * (1-face_likeliness):.2f}% confidence.")


root=Tk()
root.title("FACE_Dream Inference ")
root.geometry('600x400')

lbl = Label(root, text="Hello world!")
lbl.pack()

def infere_new_image():
    lbl.configure(text="New Image")

btn = Button(root, text="Load new Image",
             fg = "red", command=infere_new_image)

btn.pack()

root.mainloop()