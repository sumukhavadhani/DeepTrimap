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

parser = argparse.ArgumentParser(description='Crops images for ML training')
parser.add_argument('input_rgb_folder_path', action="store")
parser.add_argument('input_gt_folder_path', action="store")
parser.add_argument('--crop_width', action="store", type=int, default=64)
parser.add_argument('--crop_height', action="store", type=int, default=64)
parser.add_argument('--x_stride', action="store", type=int, default=2)
parser.add_argument('--y_stride', action="store", type=int, default=2)
parser.add_argument('--pattern', action="store", default="*.*")
parser.add_argument('--max_crops_per_image', action="store", type=int, default=3000)

def load_image(filename):
  return Image.open(filename)

def crop_image(img, x, y, cw, ch):
  # timg = img.copy()
  return img.crop((x, y, cw, ch))

def save_image(img, filename):
  img.save(filename)

def get_filnames_iter(folder_path, pattern):
  return glob.iglob(folder_path + "/" + pattern)

def crop_tuples(
  image_width, 
  image_height, 
  x_stride, 
  y_stide, 
  crop_width, 
  crop_height,
  max_crop_candidates):
  crop_candidates = []
  for x in range(0, image_width-crop_width-1, x_stride):
    for y in range(0, image_height-crop_height-1, y_stide):
      crop_candidates.append((x, y, x+crop_width, y+crop_height))
  return random.sample(population=crop_candidates, k=min(max_crop_candidates,len(crop_candidates)))

def get_basename(filename):
  return basename(filename)

def get_gt_filename(input_gt_folder_path, file_basename):
  return os.path.join(input_gt_folder_path, file_basename)

def get_output_filename(output_folder_path, file_basename, file_counter):
  return os.path.join(output_folder_path, os.path.splitext(file_basename)[0] + ("_%010d" % file_counter) + ".png")

def get_gt_label(gt_img):
  gray_img = gt_img.convert('L')  
  centre_val = gray_img.getpixel((int(gray_img.width/2),int(gray_img.height/2)))
  if centre_val > 160:
    label_str = 'foreground'
    label_val = 2
  elif centre_val > 80:
    label_str = 'unknown'
    label_val = 1
  else:
    label_str = 'background'
    label_val = 0
  return label_str, label_val

def generate_crops(
  input_rgb_folder_path, 
  output_rgb_folder_path, 
  input_gt_folder_path, 
  output_gt_folder_path,
  gt_csv_filename, 
  x_stride, 
  y_stide, 
  crop_width, 
  crop_height,
  file_pattern,
  max_crop_candidates):
  file_counter = 0
  gtdf = pd.DataFrame(columns=['rgb_filename', 'gt_filename', 'label_str', 'label_val'])
  rgb_fileiter = get_filnames_iter(input_rgb_folder_path, file_pattern)
  for rgb_filename in rgb_fileiter:
    base_name = get_basename(rgb_filename)
    gt_filename = get_gt_filename(input_gt_folder_path, base_name)
    rgb_img = load_image(rgb_filename)
    gt_img = load_image(gt_filename)
    crops = crop_tuples(rgb_img.width, rgb_img.height, x_stride, y_stide, crop_width, crop_height, max_crop_candidates)
    for (x, y, cw, ch) in crops:      
      c_rgb_img = crop_image(rgb_img, x, y, cw, ch)
      c_gt_img = crop_image(gt_img, x, y, cw, ch)      
      c_rgb_filename = get_output_filename(output_rgb_folder_path, base_name, file_counter)
      c_gt_filename = get_output_filename(output_gt_folder_path, base_name, file_counter)
      file_counter += 1 
      save_image(c_rgb_img, c_rgb_filename)   
      save_image(c_gt_img, c_gt_filename)    
      label_str, label_val = get_gt_label(c_gt_img)
      gtdf = gtdf.append(
        {
          'rgb_filename': c_rgb_filename, 
          'gt_filename': c_gt_filename, 
          'label_str': label_str, 
          'label_val': label_val
        }, ignore_index=True)
      # cropped_img.close()   
      if file_counter % 1000 == 0:
        print("Saved %010d files" % (file_counter,))
    gt_img.close()
    rgb_img.close()
  gtdf.to_csv(gt_csv_filename, index=True)
  
def main():
  args = parser.parse_args()

  output_rgb_folder_path = os.path.join(
    os.path.dirname(args.input_rgb_folder_path),
    'output_rgb_%sx%s_%s_%s' % (args.crop_width, args.crop_height, args.x_stride, args.y_stride))
  
  os.mkdir(output_rgb_folder_path)

  output_gt_folder_path = os.path.join(
    os.path.dirname(args.input_gt_folder_path),
    'output_gt_%sx%s_%s_%s' % (args.crop_width, args.crop_height, args.x_stride, args.y_stride))
  
  os.mkdir(output_gt_folder_path)

  gt_csv_filename = os.path.join(
    os.path.dirname(args.input_rgb_folder_path), 
    'gt_csv_%sx%s_%s_%s.csv' % (args.crop_width, args.crop_height, args.x_stride, args.y_stride)
  )

  generate_crops(
    args.input_rgb_folder_path,
    output_rgb_folder_path, 
    args.input_gt_folder_path, 
    output_gt_folder_path,
    gt_csv_filename,
    args.x_stride,
    args.y_stride,
    args.crop_width,
    args.crop_height,
    args.pattern,
    args.max_crops_per_image)

if __name__ == "__main__":
  main()