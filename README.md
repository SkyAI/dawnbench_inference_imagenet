# dawnbench_inference_imagenet

## run inference
1. Clone this repo.
2. Put your Imagenet validation set in `val_files/`.
```Example:
./val_files/
├── ILSVRC2012_val_00000001.JPEG
├── ILSVRC2012_val_00000002.JPEG
├── ILSVRC2012_val_00000003.JPEG
├── ILSVRC2012_val_00000004.JPEG
├── ILSVRC2012_val_00000005.JPEG
├── ILSVRC2012_val_00000006.JPEG
```
3. Pull nvcr.io/nvidia/tensorrt:19.09-py3 from NGC.
4. Start nvcr.io/nvidia/tensorrt:19.09-py3 container.
5. In the container, run cmd as below:
```
# Install opencv
apt-get update
apt-get install libopencv-dev

# Assuming you mounted this repo to /dawnbench
cd /dawnbench/   

# Build
./make.sh

# Run
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
./inference engine/engine.plan val_files/ labels.txt
```
