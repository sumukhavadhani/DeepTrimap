from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import generators
from PIL import Image
import glob
import sys
import os
from os.path import basename
import argparse
import time
import random
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np

parser = argparse.ArgumentParser(description='Crops images for ML training')
parser.add_argument('rgb_filename', action="store")
parser.add_argument('model_file_dir', action="store")
parser.add_argument('--skip_crop', action="store", type=bool, default=False)
parser.add_argument('--crop_width', action="store", type=int, default=64)
parser.add_argument('--crop_height', action="store", type=int, default=64)
parser.add_argument('--x_stride', action="store", type=int, default=2)
parser.add_argument('--y_stride', action="store", type=int, default=2)
parser.add_argument('--pattern', action="store", default="*.*")
parser.add_argument('--output_csv_filename', action="store", default="")

IMAGE_SIZE = 128
NUM_CHANNELS  = 3
BATCH_SIZE    = 50
NUM_OUTPUT_LEVELS = 3

def load_image(filename):
    return Image.open(filename)


def crop_image(img, x, y, cw, ch):
    return img.crop((x, y, cw, ch))


def save_image(img, filename):
    img.save(filename)


def crop_tuples(
    image_width,
    image_height,
    x_stride,
    y_stide,
    crop_width,
    crop_height):
    crop_candidates = []
    for x in range(0, image_width - crop_width - 1, x_stride):
        for y in range(0, image_height - crop_height - 1, y_stide):
            crop_candidates.append((x, y, x + crop_width, y + crop_height))
    return crop_candidates


def get_basename(filename):
  return basename(filename)


def get_gt_filename(input_gt_folder_path, file_basename):
    return os.path.join(input_gt_folder_path, file_basename)


def get_output_filename(output_folder_path, file_basename, file_counter):
    return os.path.join(output_folder_path, os.path.splitext(file_basename)[0] + ("_%010d" % file_counter) + ".png")


def generate_crops(
    input_rgb_filename,
    output_rgb_folder_path,
    x_stride,
    y_stide,
    crop_width,
    crop_height,
    file_pattern):
    file_counter = 0
    odf = pd.DataFrame(columns=['rgb_filename', 'gt_filename', 'label_str', 'label_val'])
    base_name = get_basename(input_rgb_filename)
    rgb_img = load_image(input_rgb_filename)
    crops = crop_tuples(rgb_img.width, rgb_img.height, x_stride, y_stide, crop_width, crop_height)
    for (x, y, cw, ch) in crops:
        c_rgb_img = crop_image(rgb_img, x, y, cw, ch)
        c_rgb_filename = get_output_filename(
        output_rgb_folder_path, base_name, file_counter)
        file_counter += 1
        save_image(c_rgb_img, c_rgb_filename)
        label_val = 0
        odf = odf.append({
            'rgb_filename': c_rgb_filename,
            'x': x,
            'y': y,
            'cw': cw,
            'ch': ch,
            'label_val': label_val
            }, ignore_index=True)
        if file_counter % 1000 == 0:
            print("Saved %010d files" % (file_counter,))
    rgb_img.close()
    return odf

def load_data(data_df):    
    filenames = np.asarray(data_df.rgb_filename.tolist()) 
    return filenames

def run_prediction(input_df, model_meta_file):
    all_filepaths = load_data(input_df)
    all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
    input_queue = tf.train.slice_input_producer([all_images],shuffle=False)
    file_content = tf.read_file(input_queue[0])
    test_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
    test_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    test_batch = tf.train.batch([test_image], batch_size=BATCH_SIZE)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.import_meta_graph(model_meta_file)
        saver.restore(sess,'/Users/sumukhavadhani/work/datasets/models-5000')

        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)
        
        x = tf.get_collection('x')[0]
        y_conv = tf.get_collection('y_conv')[0]
        keep_prob = tf.get_default_graph().get_tensor_by_name('dropout/Placeholder:0')
                 
        y_conv_out = sess.run(y_conv, feed_dict = {x: sess.run(test_batch), keep_prob: 1.0})
        print(y_conv_out)

        # for i in range(10):
        #     print(sess.run(test_batch))

        coord.request_stop()
        coord.join(threads)
        sess.close()




def main():
    args = parser.parse_args()

    if args.skip_crop is False:
        output_rgb_folder_path = os.path.join(
        os.path.dirname(args.rgb_filename),
        'crop_rgb_%sx%s_%s_%s' % (args.crop_width, args.crop_height, args.x_stride, args.y_stride))

        os.mkdir(output_rgb_folder_path)

        output_csv_filename = os.path.join(
            os.path.dirname(args.rgb_filename),
            'crop_csv_%sx%s_%s_%s.csv' % (
                args.crop_width, args.crop_height, args.x_stride, args.y_stride)
        )
    
        print('Cropping image...')
        odf = generate_crops(
            args.rgb_filename,
            output_rgb_folder_path,
            args.x_stride,
            args.y_stride,
            args.crop_width,
            args.crop_height,
            args.pattern)
        print('Cropping done...')
        odf.to_csv(output_csv_filename, index=True)
    else:
        odf = pd.read_csv(args.output_csv_filename)

    run_prediction(odf, args.model_file_dir)


if __name__ == "__main__":
    main()

