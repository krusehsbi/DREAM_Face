from FaceIdentifier import infer_images
from utils import *
import numpy as np
from keras import saving, ops, applications
from tkinter import Tk, Label, PhotoImage, Button
from PIL import Image, ImageTk

if __name__ == "__main__":
    # Flags to control data serialization and deserialization
    DESERIALIZE_DATA = True  # If True, attempt to load preprocessed data from a serialized blobs
    SERIALIZE_DATA = True  # If True, save the preprocessed data for future use

    # Load a pre-trained model from the specified file
    model = saving.load_model('saved_models/Face.keras')
    model.summary()  # Print a summary of the model's architecture

    # Define directories for face and non-face image data
    face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
    non_face_directory = ['data/nonface']

    # Load images and their labels from the specified directories
    # `deserialize_data` will try to load previously serialized data if available
    # `serialize_data` will save the loaded and preprocessed data for future runs
    images, labels = load_data(face_directory, non_face_directory, deserialize_data=DESERIALIZE_DATA,
                               serialize_data=SERIALIZE_DATA, preprocess_fnc=None)
    # Shuffle the images and labels to ensure randomized input order
    images, labels = shuffle_arrays(images, labels)

    # Create the main application window
    root = Tk()
    root.title("FACE_Dream Inference")  # Set the window title
    root.geometry('600x400')  # Define the window dimensions

    # Create a label to display instructions
    lbl = Label(root, text="Press the button to load an image.")
    lbl.pack()  # Add the label to the window

    # Create a label that will display the image
    image_label = Label(root)
    image_label.pack(fill="both", expand=True)  # Expand the label to fill available space

    # Global variable to store a reference to the current displayed image
    imgtk_ref = None

    def infere_new_image():
        """
            Function to randomly select an image from the dataset, run inference on it,
            and display the image with its predicted label in the application window.
        """
        # Randomly select an image from the dataset
        test_image = images[np.random.choice(images.shape[0], 1, replace=False)]

        # Run inference on the selected image using the pre-trained model
        label = infer_images(applications.efficientnet.preprocess_input(test_image), model, False)

        # Get the dimensions of the image container for resizing
        container_height = image_label.winfo_height()

        # Resize the selected image to fit the display container
        im = Image.fromarray(test_image[0])
        im_resized = im.resize((int(container_height * 0.8), int(container_height * 0.8)))
        imgtk = ImageTk.PhotoImage(im_resized)

        # Update the global reference to prevent image garbage collection
        global imgtk_ref
        imgtk_ref = imgtk

        # Update the label text with the predicted label and display the resized image
        lbl.configure(text=label)
        image_label.configure(image=imgtk)
        image_label.image = imgtk  # Keep a reference to prevent garbage collection

    # Create a button to trigger the loading of a new image and inference
    btn = Button(root, text="Load new Image",
                 fg="red", command=infere_new_image)
    btn.pack()  # Add the button to the window

    # Start the application's main event loop
    root.mainloop()