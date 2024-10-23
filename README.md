# DREAM_Face
DREAM-Face: Deep Regression for Estimating Age and Metrics from Face Data

## Dependencies
Place the pictures from [UTKFace](https://susanqq.github.io/UTKFace/) into subdirectory **data/utk-face**

## Setup
```
pip install --upgrade pip
```
```
pip install tensorflow[and-cuda]==2.16.1
```
```
pip install keras
```
```
pip install scikit-learn
```
```
pip install matplotlib
```
```
pip install opencv-python
```
### Verify the installation worked
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
#### !! IF THE INSTALLATION DID NOT WORK AND THE GPU IS NOT RECOGNIZED TRY THIS !!
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