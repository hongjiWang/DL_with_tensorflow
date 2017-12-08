"""
    Bullds the mnist network
    Implements the inference/loss/training pattern for model building
    1, inference() Builds the model as far as required for running the network
    forward to make predictions
    2. loss()  Adds to the inference model the layers required to generate loss
    3. training()  Adds to the loss model the Ops required to generate and apply
    gradients.

    This file is used by various "fully_connected_*.py" files and not meant to
    be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# the mnist dataset has 10 classes, representing the digits 0 throught 9
NUM_CLASSES = 10

# the mnist images are always 28 X 28 pixels
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units):
    """
    Builds the mnist model up to where it may be used to inference.
    :param images: Images placeholder, from inputs();
    :param hidden1_units: Size of the first hidden layer
    :param hidden2_units: Size of the second hidden layer.
    :return:
        softmax_linear:  Output tensor with the computed logits
    """
    # hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                                  stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                              name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # hidden2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                                                  stddev= 1.0 / math.sqrt(float(hidden1_units))),
                              name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1,weights) + biases)
    # linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                                  stddev=1.0 / math.sqrt(float(hidden2_units))),
                              name='weight')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def loss(logits, labels):
    """
    Calculates the loss from the logits and the labels
    :param logits: Logits tensor, float [batch_size, NUM_CLASSES]
    :param labels: Labels tensor, int32, [batch_size]
    :return:
        loss: Loss tensor for type float
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy')

def training(loss, learning_rate):
    """
    :param loss:  Loss tensor, from loss()
    :param learning_rate:  learning rate to use for gradient descent
    :return:
        train_op: the Op for training
    """
    # add a scalar summary for a snapshot loss
    tf.summary.scalar('loss', loss)
    # create the gradient descent optimizer with the learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # use the optimizer to apply the gradient that minimize the loss
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    """
    evaluate the quality of the logits at predicting the label
    :param logits: Logits tensor, flaot [batch_size, NUM_CLASSES]
    :param labels: Labels tensor, int32 [batch_size]  with values in
    range [0, NUM_CLASSES]
    :return:
        A scalar int32 tensor with the number of examples(out of batch_sizze)
        that were predicted correctly
    """
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))