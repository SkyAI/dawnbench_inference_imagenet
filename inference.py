# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import numpy as np
import itertools

PB_INPUT = 'input_tensor'
PB_OUTPUTS = ['softmax_tensor:0']

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

_RESIZE_MIN = 256
INPUT_SIZE = 224
INPUT_DIMENSIONS = (INPUT_SIZE, INPUT_SIZE)
NUM_CLASSES = 1001

TEST_COUNT=50000
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'


def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)
    return image - means


def _smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)
    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    return tf.image.resize_images(
        image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)


def preprocess_image(image_buffer,
                     output_height,
                     output_width,
                     num_channels=3
                     ):
    # For validation, we want to decode, resize, then just crop the middle.
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels, dct_method='INTEGER_FAST')
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = _central_crop(image, output_height, output_width)
    image.set_shape([output_height, output_width, num_channels])

    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)

def deserialize_image_record(record):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        return imgdata, label


def get_preprocess_fn(model, mode='classification'):
    def process(record):
        imgdata, label = deserialize_image_record(record)
        image = preprocess_image(imgdata, INPUT_SIZE, INPUT_SIZE)
        return image, label

    return process

class LoggerHook(tf.train.SessionRunHook):
    """Logs runtime of each iteration"""

    def __init__(self, batch_size, num_records, display_every):
        self.iter_times = []
        self.display_every = display_every
        self.num_steps = (num_records + batch_size - 1) / batch_size
        self.batch_size = batch_size

    def begin(self):
        self.start_time = time.time()

    def before_run(self, run_context):
        self.start_time = time.time()

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self.start_time
        self.iter_times.append(duration)
        current_step = len(self.iter_times)
        if current_step % self.display_every == 0:
            print("    step %d/%d, iter_time(ms)=%.4f, images/sec=%d" % (
                current_step, self.num_steps, duration * 1000,
                self.batch_size / self.iter_times[-1]))

def run(frozen_graph, model, data_files, batch_size,
        num_warmup_iterations=100, display_every=100, intra=1, inter=8):
    # Define model function for tf.estimator.Estimator
    
    def model_fn(features, labels, mode):
        logits_out = tf.import_graph_def(frozen_graph,
                                         input_map={'input_tensor': features},
                                         return_elements=['softmax_tensor:0'],
                                         name='')

        predictions = {
            'probabilities': logits_out[0],
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions)

    # preprocess function for input data
    preprocess_fn = get_preprocess_fn(model)

    def get_tfrecords_count(files):
        num_records = 0
        for fn in files:
            for record in tf.python_io.tf_record_iterator(fn):
                num_records += 1
        return num_records

    # Define the dataset input function for tf.estimator.Estimator
    def eval_input_fn():
        dataset = tf.data.TFRecordDataset(data_files)
        dataset = dataset.cache()
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_calls=16))
        dataset = dataset.prefetch(4)
        dataset = dataset.repeat(count=1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    # Evaluate model
    num_records=get_tfrecords_count(data_files)
    logger = LoggerHook(
        display_every=display_every,
        batch_size=batch_size,
        num_records=get_tfrecords_count(data_files))
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads = intra
    tf_config.inter_op_parallelism_threads = inter
    
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(session_config=tf_config),
        model_dir='model_dir')
    
    results = estimator.predict(eval_input_fn, yield_single_examples=True, hooks=[logger])
    predictions = list(itertools.islice(results, num_records))

    # Gather additional results
    iter_times = np.array(logger.iter_times[num_warmup_iterations:])
    mean_time = np.mean(iter_times) * 1000

    return predictions, mean_time

def get_frozen_graph(prebuilt_graph_path):
    
    num_nodes = {}
    times = {}
    graph_sizes = {}

    # Load from pb file if frozen graph was already created and cached
    if os.path.isfile(prebuilt_graph_path):
        print('Loading cached frozen graph from \'%s\'' % prebuilt_graph_path)
        start_time = time.time()
        with tf.gfile.GFile(prebuilt_graph_path, "rb") as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        times['loading_frozen_graph'] = time.time() - start_time
        num_nodes['loaded_frozen_graph'] = len(frozen_graph.node)
        num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op) == 'TRTEngineOp'])
        graph_sizes['loaded_frozen_graph'] = len(frozen_graph.SerializeToString())
        return frozen_graph, num_nodes, times, graph_sizes

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, required=True,
                        help='Model file name')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing validation set TFRecord files.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of images per batch.')
    parser.add_argument('--display_every', type=int, default=1000,
                        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--num_warmup_iterations', type=int, default=100,
                        help='Number of initial iterations skipped from timing')
    parser.add_argument('--intra', type=int, default=1)
    parser.add_argument('--inter', type=int, default=1)
    args = parser.parse_args()

    def get_files(data_dir, filename_pattern):
        if data_dir == None:
            return []
        files = tf.gfile.Glob(os.path.join(data_dir, filename_pattern))
        if files == []:
            raise ValueError('Can not find any files in {} with pattern "{}"'.format(
                data_dir, filename_pattern))
        return files

    validation_files = get_files(args.data_dir, 'validation*')

    # Retreive graph using NETS table in graph.py
    frozen_graph, num_nodes, times, graph_sizes = get_frozen_graph(
        prebuilt_graph_path=args.model
        )

    def print_dict(input_dict, str='', scale=None):
        for k, v in sorted(input_dict.items()):
            headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
            v = v * scale if scale else v
            print('{}{}'.format(headline, '%.1f' % v if type(v) == float else v))

    print_dict(vars(args))
    print_dict(num_nodes, str='num_nodes')
    print_dict(graph_sizes, str='graph_size(MB)', scale=1. / (1 << 20))
    print_dict(times, str='time(s)')

    # Predict model
    print('running inference...')
    predictions, mean_time = run(
        frozen_graph,
        model=args.model,
        data_files=validation_files,
        batch_size=args.batch_size,
        num_warmup_iterations=args.num_warmup_iterations,
        display_every=args.display_every,
        intra=args.intra,
        inter=args.inter)

    # Calculate Accuracy
    print('calculating accuracy...')
    top5classes = []
    for i in range(TEST_COUNT):
        top5classes.append(predictions[i]['probabilities'].argsort()[-5:][::-1])

    g_1 = tf.Graph()
    with g_1.as_default():
        dataset = tf.data.TFRecordDataset(validation_files)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=get_preprocess_fn(args.model), batch_size=1, num_parallel_calls=8))
        dataset = dataset.prefetch(buffer_size=4)
        dataset = dataset.repeat(count=1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
    
    predict_top_5_true = 0
    with tf.Session(graph=g_1) as sess:
        for i in range(TEST_COUNT):
            results = sess.run(labels)
            if results[0][0] in top5classes[i]:
                predict_top_5_true += 1

    accuracy = float(predict_top_5_true) / TEST_COUNT
    print('results of {}:'.format(args.model))
    print('    top5 accuracy: %.2f' % (accuracy * 100))
    print('    latency_mean(ms): %.4f' % mean_time)
