# DREAM_Face
DREAM-Face: Deep Regression for Estimating Age and Metrics from Face Data

## Dependencies
Place the pictures from [UTKFace](https://susanqq.github.io/UTKFace/) into subdirectory **data/utk-face**

## Setup
### Option 1: install script (GPU and CPU)
Execute the install-dependencies.sh script in the root directory of your project.

### Option 2: requirements.txt (ONLY CPU)
Install all the requirements from the requirements.txt in case you are only using CPU.

### Option 3: requirements-yourPlatform.txt (CPU AND GPU)
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
#### !! IF THE INSTALLATION DID WORK FOR CUDA BUT THE GPU IS NOT RECOGNIZED TRY THIS !!
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