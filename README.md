# DREAM_Face
DREAM-Face: Deep Regression for estimating Age and Metrics from Face Data.

It uses roughly 50000 UTK-Face and openimagesV7 images to train and achieves

Face Presence Accuracy:     98% 

Age MAE:                    8.2

Gender Accuracy:            93.5%

It uses EfficientNetB0 at its core and is therefore very small and lightweight.

## Dependencies
## Setup
### Option 1: requirements.txt (ONLY CPU)
Install all the requirements from the requirements.txt in case you are only using CPU.

### Option 2: requirements-yourPlatform.txt (CPU AND GPU)
Select the requirements file you want to use for your specific hardware and execute this command

```
pip install -r requirements-yourPlafrom.txt
```
e.g.
```
pip install -r requirements-cuda.txt   # for NVIDIA
pip install -r requirements-rocm.txt   # for AMD
pip install -r requirements-cpu.txt    # for CPU
```

### Verify the installation worked
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
### !! IF THE INSTALLATION DID WORK FOR CUDA BUT THE GPU IS NOT RECOGNIZED TRY THIS !!
* Create symbolic links to NVIDIA shared libraries:
```
pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd
```
* Create a symbolic link to ptxas:
```
ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc;         
print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas
```
Verify the GPU setup:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

# Dataset Download
To download the dataset execute this command:
```
python3 Dataset-Downloader.py
```

## Running the different models
After you've setup all the dependencies you can now use the models.
#### !! RUNNING ONE OF THE MODEL FILES FOR THE FIRST TIME WILL TAKE A LOT OF TIME SINCE THE WHOLE DATABASE WILL BE LOEADED AND THEN SERIALIZED. AFTER THAT IT WILL BE MUCH FASTER !!

## Predictions using viewer.py
This script can be used to generate a total of 6 predictions with 3 faces and 3 non faces and display the result and/or
save it to a file. If you execute the script without arguments you will get a help menu that describes how to use it.

## Prediction using FaceInference.py
This script can be used to predict a single random image from the database using a simple window.

### Face Detector
This model is used to detect whether a face is in a picture or not.
There is a finished model under ```saved_models/FaceDetektor.keras``` which can be used if you don't want to train
the model yourself.

### Face
This model is used to detect whether a face is in a picture and if so how old the person is and what gender they have.
There is a finished model under ```saved_models/Face.keras``` which can be used if you don't want to train the model 
yourself.