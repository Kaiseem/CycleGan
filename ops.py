import tensorflow as tf
import math

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def conv(inputs, kernel_size, output_num,strides,name, padding='SAME',weight_decay=0.0001,act_func=None):
    with tf.variable_scope(name):
        input_size = inputs.get_shape().as_list()[-1]
        conv_weights = tf.get_variable(name='weights',
            initializer=tf.truncated_normal([kernel_size, kernel_size, input_size, output_num], dtype=tf.float32, stddev=math.sqrt(2 / (kernel_size * kernel_size * output_num))),
            )
        #conv_biases = tf.get_variable( 'biases',initializer=tf.constant(0.0, shape=[output_num], dtype=tf.float32))
        conv_layer = tf.nn.conv2d(inputs, conv_weights, [1, strides, strides, 1], padding=padding)
        #conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
        #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(weight_decay)(conv_weights))
        if act_func:
            conv_layer = act_func(conv_layer)
    return conv_layer

