import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

import tensorflow as tf

print(tf.__version__)

print(keras.version())