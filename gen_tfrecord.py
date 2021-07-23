import os
import pandas as pd
import csv

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)



if not os.path.exists('/home/shyam/bridge_tech/new_view/fin_dataset/data/validation'):
    os.makedirs('/home/shyam/bridge_tech/new_view/fin_dataset/data/validation')

if not os.path.exists('/home/shyam/bridge_tech/new_view/fin_dataset/data/train'):
    os.makedirs('/home/shyam/bridge_tech/new_view/fin_dataset/data/train')

# def write_tfrecords(x, y, filename):
#     writer = tf.io.TFRecordWriter(filename)

#     for image, label in zip(x, y):
#         example = tf.train.Example(features=tf.train.Features(
#             feature={
#                 'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
#                 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
#             }))
#         writer.write(example.SerializeToString())

# write_tfrecords(x_test, y_test, './data/validation/validation_30k_14_6.tfrecords')

# write_tfrecords(x_train, y_train, './data/train/train_30k_14_6.tfrecords')


from glob import glob
import os
import random

def serialize_example(image, label):

    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def make_tfrecords(path, record_file='/content/images.tfrecords'):
  classes = os.listdir(path)
  with tf.io.TFRecordWriter(record_file) as writer:
    files_list = glob(path + '/*/*')
    random.shuffle(files_list)
    for filename in files_list:
      image_string = open(filename, 'rb').read()
      category = filename.split('/')[-2]
      label = classes.index(category)
      tf_example = serialize_example(image_string, label)
      writer.write(tf_example)


make_tfrecords('/home/shyam/bridge_tech/new_view/fin_dataset/final/train','/home/shyam/bridge_tech/new_view/fin_dataset/data/train/train_30k_14_6.tfrecords')
make_tfrecords('/home/shyam/bridge_tech/new_view/fin_dataset/final/validation','/home/shyam/bridge_tech/new_view/fin_dataset/data/validation/validation_30k_14_6.tfrecords')