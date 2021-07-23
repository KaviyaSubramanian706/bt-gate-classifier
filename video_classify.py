import time
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import numpy as np
import argparse
import pandas as pd
import cv2
import csv
from sklearn.metrics import classification_report, confusion_matrix
from normalizer import Normalizer

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

stds = None
means = None

# flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './home/shyam/bridge_tech/mobilenetv2-tf2/model/ec2_model/model_1.1_15_6_64/output/',
                    'path to weights file')
flags.DEFINE_enum('model', None, ['MobileNetv2'])

flags.DEFINE_integer('size', 64, 'resize images to')
flags.DEFINE_string('video', None,
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')



def get_mean_std(base_dir, filenames, target_size):
    n = 0
    r_mean, g_mean, b_mean = 0.0, 0.0, 0.0
    r_M2, g_M2, b_M2 = 0.0, 0.0, 0.0

    
    # for z, filename in enumerate(filenames):
    #     if z % 1000 == 0:
    #         print("Processing image {}/{}".format(z+1, len(filenames)))

        # x = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(os.path.join(base_dir, filename), target_size=target_size))
    r = x[:, :, 0].flatten().tolist()
    g = x[:, :, 1].flatten().tolist()
    b = x[:, :, 2].flatten().tolist()

    for (xr, xg, xb) in zip(r, g, b):
        n = n + 1

        r_delta = xr - r_mean
        g_delta = xg - g_mean
        b_delta = xb - b_mean

        r_mean = r_mean + r_delta/n
        g_mean = g_mean + g_delta/n
        b_mean = b_mean + b_delta/n

        r_M2 = r_M2 + r_delta * (xr - r_mean)
        g_M2 = g_M2 + g_delta * (xg - g_mean)
        b_M2 = b_M2 + b_delta * (xb - b_mean)

    r_variance = r_M2 / (n - 1)
    g_variance = g_M2 / (n - 1)
    b_variance = b_M2 / (n - 1)

    r_std = np.sqrt(r_variance)
    g_std = np.sqrt(g_variance)
    b_std = np.sqrt(b_variance)

    return np.array([r_mean, g_mean, b_mean]), np.array([r_std, g_std, b_std])


class Normalizer():
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if self.mean is not None:
            img = self.center(img)
        if self.std is not None:
            img = self.scale(img)
        return img

    def center(self, img):
        return img - self.mean

    def scale(self, img):
        return img / self.std

    def set_stats(self, mean, std):
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
        

    def get_stats(self, base_dir, filenames, target_size, calc_mean=True, calc_std=True):
        print("Calculating mean and standard deviation with shape: ", target_size)
        m, s = get_mean_std(base_dir, filenames, target_size)
        if calc_mean:
            self.mean = m
            self.mean = self.mean.reshape(1, 1, 3)
            print("Dataset mean [r, g, b] = {}".format(m.tolist()))
        if calc_std:
            self.std = s
            self.std = self.std.reshape(1, 1, 3)
            print("Dataset std [r, g, b] = {}". format(s.tolist()))

        return str(m.tolist()), str(s.tolist())

def main(_argv):
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #for physical_device in physical_devices:
    #    tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.model == 'yolov3-tiny':
        model = YoloV3Tiny(FLAGS.size, classes=FLAGS.num_classes,
                anchors=yolo_tiny_anchors,masks=yolo_tiny_anchor_masks)
        model.summary()

    elif FLAGS.model == 'MobileNetv2':
        model = tf.keras.models.load_model('/home/shyam/bridge_tech/mobilenetv2-tf2/model/ec2_model/model_1.1_15_6_64/output/')
        model.summary()


    model.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = ['Open','Closed']
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            break

        img = tf.keras.preprocessing.image.load_img(
        img, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = tf.image.resize(img_in, (FLAGS.size, FLAGS.size))
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = model.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
