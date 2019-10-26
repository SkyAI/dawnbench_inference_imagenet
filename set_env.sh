#!/bin/bash

GPU_ID=0 #Change this to your GPU id
MAX_SM_CLOCK=`nvidia-smi -q -d CLOCK -i ${GPU_ID} | grep -A4 "Max Clocks" | grep SM | awk '{print $3}'`
MAX_MEM_CLOCK=`nvidia-smi -q -d CLOCK -i ${GPU_ID} | grep -A4 "Max Clocks" | grep Memory | awk '{print $3}'`
nvidia-smi -pm 1 -i ${GPU_ID}
nvidia-smi -ac ${MAX_MEM_CLOCK},${MAX_SM_CLOCK} -i ${GPU_ID}
