from nvidia.dali.pipeline import Pipeline
from nvidia import dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import tensorflow as tf
import numpy as np
import sys, os
import time
sys.path.insert(1, os.path.join(sys.path[0], ".."))
_R_MEAN = 123.68
_G_MEAN = 116.28
_B_MEAN = 103.53
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256
INPUT_SIZE = 224
INPUT_DIMENSIONS = (INPUT_SIZE, INPUT_SIZE)
NUM_CLASSES = 1001
TEST_COUNT = 50000
DISPLAY_COUNT = 1000
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_FILE = "./resnet50.model"
DATA_DIR = "./val_files"
IDX_DIR = "./idx_files"
BATCH_SIZE = 1
DTYPE = trt.float32

class Dali_CPU_Pipe(Pipeline):
    def __init__(self, tfrec_filenames, tfrec_idx_filenames, batch_size, num_threads, device_id, set_affinity, prefetch_queue_depth):
        super(Dali_CPU_Pipe, self).__init__(batch_size, num_threads, device_id, set_affinity=set_affinity, prefetch_queue_depth=prefetch_queue_depth)
        self.input = ops.TFRecordReader(path = tfrec_filenames,
                                        index_path = tfrec_idx_filenames,
                                        initial_fill=10000,
                                        features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                                    'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64,  -1)})
        self.decode = ops.HostDecoder(output_type = types.RGB)
        self.resize = ops.Resize(device = "cpu", resize_shorter = _RESIZE_MIN)
        self.cmnp = ops.CropMirrorNormalize(device = "cpu", output_dtype = types.FLOAT, crop = (INPUT_SIZE, INPUT_SIZE), image_type = types.RGB, mean = _CHANNEL_MEANS, std = [58.395, 57.120, 57.375], output_layout=dali.types.NCHW)
        self.iter = 0

    def define_graph(self):
        inputs  = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"].cpu()
        images = self.decode(images)
        images = self.resize(images)
        images = self.cmnp(images.cpu())
        return (images, labels)
    def iter_setup(self):
        pass

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

def get_files(data_dir, filename_pattern):
    if data_dir == None:
        return []
    files = tf.gfile.Glob(os.path.join(data_dir, filename_pattern))
    if files == []:
        raise ValueError('Can not find any files in {} with pattern "{}"'.format(
            data_dir, filename_pattern))
    return files

def dali_pipe_init(data_dir, idx_dir):
    files = sorted(get_files(data_dir, 'validation*'))
    idx_files = sorted(get_files(idx_dir, 'validation*'))
    pipe = Dali_CPU_Pipe(tfrec_filenames=files, tfrec_idx_filenames=idx_files, batch_size=BATCH_SIZE, num_threads=1, device_id = 0, set_affinity=True, prefetch_queue_depth={"cpu_size":1, "gpu_size":1})
    return pipe

def get_list(tensor_lists):
    return tensor_lists.at(0)
 
    

def main():
    pipe = dali_pipe_init(data_dir = DATA_DIR, idx_dir = IDX_DIR)
    pipe.build()
    with open(ENGINE_FILE, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            top5classes = []
            labelclasses = []
            timeclasses = []
            i = 0
            while i < TEST_COUNT:
                images, labels = pipe.run()
                np.copyto(h_input, np.asarray(get_list(images)).flatten())
                start_time = time.time()
                result = do_inference(context, h_input, d_input, h_output, d_output, stream)
                end_time = time.time()
                top5classes.append(result.argsort()[-5:][::-1])
                labelclasses.append(get_list(labels)[0])
                timeclasses.append(end_time-start_time)
                i = i + 1
                if (i % DISPLAY_COUNT == 0):
                    print(i,"images inference avg latency time = %.4f ms" % (sum(timeclasses[-DISPLAY_COUNT:]) * 1000.0 / DISPLAY_COUNT))
            
            predict_top_5_true = 0
            for i in range(TEST_COUNT):
                if labelclasses[i]-1 in top5classes[i]:
                    predict_top_5_true += 1
            accuracy = float(predict_top_5_true) / TEST_COUNT
            print('    accuracy: %.2f' % (accuracy * 100))
            print("all images inference avg latency time = %.4f ms" % (sum(timeclasses) * 1000.0 / TEST_COUNT))

if __name__ == '__main__':
    main()

