from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# import keras
import numpy as np
import argparse
import os, sys
import json
import glob
import random
import collections
import math
import time
from google.protobuf import text_format

# parser = argparse.ArgumentParser()
# parser.add_argument("--output_dir", required=True, help="where to put output files")
# parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
# parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
# parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
# parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
# parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
# args = parser.parse_args()
#
# output_dir = args.output_dir
# trace_freq = args.trace_freq
# summary_freq = args.summary_freq
# checkpoint = args.checkpoint
# separable_conv = args.separable_conv
# ngf = args.ngf

output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches_out"
trace_freq = 100
summary_freq = 100
checkpoint = ""
separable_conv = True
ngf = 64


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          kernel_initializer=initializer)


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, ngf)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


model_fn = "/infodev1/non-phi-data/junjiang/OvaryCancer/penMarking/trained_model/graph.pbtxt"
saved_model_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/penMarking/trained_model"

f = open(model_fn, "r")
graph_protobuf = text_format.Parse(f.read(), tf.compat.v1.GraphDef())
tf.import_graph_def(graph_protobuf, name='')
writer = tf.summary.FileWriter("/tmp/mylogs")
writer.add_summary(tf.compat.v1.summary.initialize(graph_protobuf))
writer.close()

graph_nodes = [n for n in graph_protobuf.node]

new_model = tf.keras.models.load_model(model_fn)

model = tf.saved_model.load(saved_model_dir)

with open(model_fn, "r") as f:
    graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())
print("debug")
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

x = graph.get_tensor_by_name('input_tensor:0')
y = graph.get_tensor_by_name("StatefulPartitionedCall/Postprocessor/Tile/multiples:0")

x_val = np.random.rand(1, 300, 300, 3).astype("float32")
# x_val = np.ones(shape=(1,300,300,3), dtype="uint8")

with tf.compat.v1.Session(graph=graph) as sess:
    y_out = sess.run(y, feed_dict={x: x_val})

print("output shape", y_out.shape)

# case_patches_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256/OCMC-006"
case_patches_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224_norm/OCMC-017/"
sess = tf.compat.v1.Session()
with tf.variable_scope("generator"):
    input_paths = glob.glob(os.path.join(case_patches_dir, "*.png"))

    if os.path.splitext(input_paths[0])[1] == ".jpg":
        decode = tf.image.decode_jpeg
    elif os.path.splitext(input_paths[0])[1] == ".png":
        decode = tf.image.decode_png
    else:
        raise Exception("Image files can't be decoded")

    path_queue = tf.train.string_input_producer(input_paths, shuffle=False)
    reader = tf.WholeFileReader()
    paths, contents = reader.read(path_queue)

    raw_input = decode(contents)
    raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

    assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_input)

    raw_input.set_shape([None, None, 3])
    width = tf.shape(raw_input)[1]  # [height, width, channels]
    a_images = preprocess(raw_input[:, :width // 2, :])
    b_images = preprocess(raw_input[:, width // 2:, :])

    inputs, targets = [a_images, b_images]

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    tf.print(width, output_stream=sys.stderr)
    print("debug")
    # print("parameter_count =", sess.run(parameter_count))

    print("image width =", sess.run(width))

    sess.close()
    print("debug")

    # out_channels = int(targets.get_shape()[-1])
    # gen_model = create_generator(inputs, out_channels)
    #
    # saver = tf.train.Saver(max_to_keep=1)
    #
    # logdir = output_dir if (trace_freq > 0 or summary_freq > 0) else None
    # sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    #
    # with sv.managed_session() as sess:
    #     print("parameter_count =", sess.run(parameter_count))

    # if checkpoint is not None:
    #     print("loading model from checkpoint")
    #     checkpoint = tf.train.latest_checkpoint(checkpoint)
