"""trains and evaluates the mnist network using a feed dictionary"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

from six.moves import xrange
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# basic model parameter as a external flags
FLAGS = None

def placeholder_inputs(batch_size):
    """
    Generate the placeholder variables to represent input tensor
    :param batch_size: the batch size will be baked into both placeholders
    :return:
        images_placeholder: Image placeholder
        labels_placeholder: Label placeholder
    """
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, label_pl):
    """
    Fills the feed_dict for training the given step
    :param data_set: The set of images and labels, from input_data.read_data_sets()
    :param images_pl: The image placeholder, from placeholder_inputs()
    :param label_pl: The label placeholder, from placeholder_inputs()
    :return:
        feed_dict: The feed dictionary mapping the placeholder to the values
    """
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        label_pl: labels_feed,
    }
    return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    """
    Runs one evaluation against the full epoch of data
    :param sess: The session in which the model has been trained
    :param eval_correct: The tensor that return the number of correct prediction
    :param images_placeholder: The images placeholder
    :param labels_placeholder: The labels placeholder
    :param data_set: The set of images and labels to evaluate, from
    input_data.read_data_sets()
    :return:
    """
    # And run a epoch of eval
    true_count = 0 # counts the number of correct predictions
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size   # 双斜杠表示向下取整
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print("Num examples: %d  Num correct: %d  precision @ 1: %0.04f" %
          (num_examples, true_count, precision))

def run_training():
    """Train the mnist for a number of steps"""
    # Get the set of images and labels for training, validation, test on mnist
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    # Tell the tensorflow that the model will be built into the default Graph
    with tf.Graph().as_default():
        # generate the placeholders for the images and labels
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a graph that computes the precisions for the inference model
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        # add the graph the Ops for loss calculation
        loss = mnist.loss(logits, labels_placeholder)

        # add the graph the Ops that calculate and apply the gradient
        train_op = mnist.training(loss, FLAGS.learning_rate)

        # evaluation
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        #build the summary tensor
        summary = tf.summary.merge_all()

        # add the variable initializer Op.
        init = tf.global_variables_initializer()
        #create a saver for writing the training checkpoint
        saver = tf.train.Saver()
        #create a session for running the Ops on the graph
        sess = tf.Session()
        # initiate a SummaryWriter to output summaries and the graph
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # and then after everything is built
        # runs Op to initialize the variable
        sess.run(init)
        # start the training loop
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            #fill the dictionary with the actual sets of images and labels
            # for this particular training step
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            # run one step of the model
            _, loss_value = sess.run([train_op, loss],
                                      feed_dict=feed_dict)
            duration = time.time() - start_time

            # write a summaries and print an overview fairly often.
            if step % 100 == 0:
                # print status to stdout
                print('Step %d: loss = %.2f (%.3f sec)'%(step, loss_value, duration))
                # update the event file
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # save a checkpoint and evaluate the model periodically
            if (step + 1) % 1000 ==0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess,checkpoint_file, global_step=step)
                # evaluate against the training set
                print("Training data Eval: ")
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # evaluate against the validation set
                print('Validation data Eval: ')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # evaluate against the test set
                print('Test data Eval: ')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial the learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR',''),
                             'input_data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', ''),
                             'logs'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



