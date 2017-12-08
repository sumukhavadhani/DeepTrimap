from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import tensorflow as tf

import numpy as np
import pandas as pd
import random
import sys


FLAGS = None

IMAGE_SIZE = 128
NUM_CHANNELS  = 3
BATCH_SIZE    = 50
NUM_OUTPUT_LEVELS = 3

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def deepnn(x):
  # Reshape to use within a convolutional neural net.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([int(IMAGE_SIZE/4) * int(IMAGE_SIZE/4) * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, NUM_OUTPUT_LEVELS])
    b_fc2 = bias_variable([3])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

def load_data(data_df):    
    filenames = np.asarray(data_df.rgb_filename.tolist()) 
    labels = np.asarray(data_df.label_val.tolist())    
    return filenames, labels

def main(_):
  # Import data
  num_epochs = 1
  input_df = pd.read_csv(FLAGS.data_dir)

  all_filepaths, all_labels = load_data(input_df)
  all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
  all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

  train_size = int(input_df.shape[0]*0.7)
  test_size = input_df.shape[0] - train_size
  partitions = [0] * input_df.shape[0]
  partitions[:test_size] = [1] * test_size
  random.shuffle(partitions)

  train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
  train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

  train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_labels],
                                    shuffle=False)
  test_input_queue = tf.train.slice_input_producer(
                                    [test_images, test_labels],
                                    shuffle=False)

  file_content = tf.read_file(train_input_queue[0])
  train_image = tf.image.decode_png(file_content)
  train_label = tf.one_hot(train_input_queue[1], NUM_OUTPUT_LEVELS)

  file_content = tf.read_file(test_input_queue[0])
  test_image = tf.image.decode_png(file_content)
  test_label = tf.one_hot(test_input_queue[1], NUM_OUTPUT_LEVELS)

  train_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
  test_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

  # collect batches of images before processing
  train_batch = tf.train.batch([train_image, train_label], batch_size=BATCH_SIZE)
  test_batch = tf.train.batch([test_image, test_label], batch_size=BATCH_SIZE)

  print("input pipeline ready")

  # Create the model
  x = tf.placeholder(tf.float32, [BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_OUTPUT_LEVELS])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      step = 0
      while not coord.should_stop():                
        if step % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: sess.run(train_batch[0]), y_: sess.run(train_batch[1]), keep_prob: 1.0})
          print('step %d, training accuracy %g' % (step, train_accuracy))
        train_step.run(feed_dict={x: sess.run(train_batch[0]), y_: sess.run(train_batch[1]), keep_prob: 0.5})
        step += 1
        if step % 1000 == 0:
          test_accuracy = accuracy.eval(feed_dict={ x: sess.run(test_batch[0]), y_: sess.run(test_batch[1]), keep_prob: 1.0})
          print('test accuracy %g' % (step, test_accuracy))
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (num_epochs, step))
      save_path = saver.save(sess, FLAGS.model_dir, global_step=step)
      print("Model saved in file: %s" % save_path)      
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str, help='Directory of input data')
  parser.add_argument('model_dir', type=str, help='Directory for storing model')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)