import os
import pandas as pd
import csv

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)



def read_dataset(epochs, batch_size, channel, channel_name):

    filenames = [os.path.join(channel, channel_name + '.tfrecords')]
    dataset = tf.data.TFRecordDataset(filenames)

    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    dataset = dataset.map(_parse_image_function, num_parallel_calls=10)
    dataset = dataset.prefetch(10)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset