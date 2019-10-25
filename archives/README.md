# dawnbench_inference_imagenet

## run inference
1. Clone this repo.
2. Put your Imagenet validation set in `val_files` directory (TFRecord).
3. Pull nvcr.io/nvidia/tensorrt:19.05-py2 from NGC.
4. Start nvcr.io/nvidia/tensorrt:19.05-py2 container.
5. In the container, run cmd as below:
```
/opt/tensorrt/python/python_setup.sh
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali-tf-plugin
cd /dawnbench/   # Assuming you mounted this repo to /dawnbench
taskset -c 2 python inference_v2.py
```
