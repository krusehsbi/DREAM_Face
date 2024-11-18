import PIL
import matplotlib.pyplot as plt
from PIL.Image import fromarray

from FaceIdentifier import infer_images
from utils import *
import numpy as np
from keras import saving, ops, applications
from tkinter import Tk, Label, PhotoImage, Button
from PIL import Image, ImageTk

if __name__ == "__main__":
    DESERIALIZE_DATA = True
    SERIALIZE_DATA = True

    model = saving.load_model('saved_models/Face.keras')
    model.summary()

    # Define directories
    face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
    non_face_directory = ['data/nonface']

    # Load images and labels from both face and non-face directories
    images, labels = load_data(face_directory, non_face_directory, deserialize_data=DESERIALIZE_DATA,
                               serialize_data=SERIALIZE_DATA, preprocess_fnc=None)
    images, labels = shuffle_arrays(images, labels)


    root=Tk()
    root.title("FACE_Dream Inference ")
    root.geometry('600x400')

    lbl = Label(root, text="Press the button to load an image.")
    lbl.pack()

    # Create a label for the image display
    image_label = Label(root)
    image_label.pack(fill="both", expand=True)

    # To store the reference to the image
    imgtk_ref = None

    def infere_new_image():
        test_image = images[np.random.choice(images.shape[0], 1, replace=False)]
        label = infer_images(applications.efficientnet.preprocess_input(test_image), model, False)

        # Get the dimensions of the image container
        container_height = image_label.winfo_height()

        # Resize the image to fit the container
        im = Image.fromarray(test_image[0])
        im_resized = im.resize((int(container_height * 0.8), int(container_height * 0.8)))
        imgtk = ImageTk.PhotoImage(im_resized)

        # Update the reference to prevent garbage collection
        global imgtk_ref
        imgtk_ref = imgtk
        # Update the label text and image
        lbl.configure(text=label)
        image_label.configure(image=imgtk)
        image_label.image = imgtk


    btn = Button(root, text="Load new Image",
                 fg = "red", command=infere_new_image)

    btn.pack()

    root.mainloop()