# dawnbench_inference_imagenet

## run inference
1. Clone this repo.
2. Place your Imagenet validation set into `dataset`. (TFRecord)
3. Pull nvcr.io/nvidia/tensorrt:19.05-py2 from NGC.
4. Start nvcr.io/nvidia/tensorrt:19.05-py2 container.
5. In the container, run cmd as below:
```
cd /opt/tensorrt/python
./python_setup.sh
wget http://developer.download.nvidia.com/compute/redist/cuda/10.0/nvidia-dali/nvidia_dali-0.10.0-743881-cp27-cp27mu-manylinux1_x86_64.whl
pip install nvidia_dali-0.10.0-743881-cp27-cp27mu-manylinux1_x86_64.whl
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali-tf-plugin
cd /dawnbench/   # Assuming you mounted this repo to /dawnbench
taskset -c 2 python inference_v2.py
```
