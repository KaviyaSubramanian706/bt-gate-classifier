import os
import numpy as np
import argparse
import pandas as pd
import csv
from sklearn.metrics import classification_report, confusion_matrix
from normalizer import Normalizer
import tensorflow_hub as hub

from mobilenetv2 import MobileNetV2
# from tensorflow.keras.applications import ResNet50, VGG16, VGG19, MobileNetV2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, MaxPool2D, MaxPooling2D



import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

stds = None
means = None

def generate(test_dir):
    """Data generation and augmentation

    # Arguments
        args: Dictionary, command line arguments
        
    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        num_training: Integer, number of images in the train split.
        num_validation: Integer, number of images in the validation split.
    """

    #  Using the data Augmentation in traning data
    
    normalizer = Normalizer()

    test_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        #rescale=1. / 255.,
        # shear_range=0.2,#args.shear_range,
        # zoom_range= 0.2,#args.zoom_range,
        # rotation_range= 90,# args.rotation_range,
        # width_shift_range=0.2, #args.width_shift_range,
        # height_shift_range= 0.2, #args.height_shift_range,
        # horizontal_flip=True, #args.horizontal_flip,
        # vertical_flip= False, # args.vertical_flip,
        preprocessing_function=normalizer)


    # validation_aug = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=normalizer)

    test_generator = test_aug.flow_from_directory(
        test_dir,
        target_size=(128,128), #(args.input_size, args.input_size),
        batch_size= 1, # args.batch_size,
        class_mode='categorical',
        shuffle=False)

    mean, std = [], []
    if means is None or stds is None:
        mean, std = normalizer.get_stats('/home/shyam/bridge_tech/new_view/fin_dataset/eval', test_generator.filenames, (128, 128))
    else:
        mean = [float(m.strip()) for m in args.mean.split(',')]
        std = [float(s.strip()) for s in args.std.split(',')]
        normalizer.set_stats(mean, std)


    return test_generator


# normalizer = Normalizer()

# test_aug = tf.keras.preprocessing.image.ImageDataGenerator(
#         #rescale=1. / 255.,
#         # shear_range=0.2,#args.shear_range,
#         # zoom_range= 0.2,#args.zoom_range,
#         # rotation_range= 90,# args.rotation_range,
#         # width_shift_range=0.2, #args.width_shift_range,
#         # height_shift_range= 0.2, #args.height_shift_range,
#         # horizontal_flip=True, #args.horizontal_flip,
#         # vertical_flip= False, # args.vertical_flip,
#         preprocessing_function=normalizer)
test_gen = generate('/home/shyam/bridge_tech/new_view/fin_dataset/eval')

# img = test_aug.load_img('/home/shyam/bridge_tech/new_view/fin_dataset/ld/27-05-2021_07-01-59.avi_ld_frame_1525.png')

model = MobileNetV2((64,64,3),2,False) #((args.input_size, args.input_size, 3), num_classes, args.plot_model)

# print('Loading MobileNetV2 ...')
# base_model = MobileNetV2(input_shape=(96,96,3),
#                 include_top=False,
#                 weights='imagenet')
# print('MobileNetV2 loaded')

# base_model.trainable = False

# model = Sequential([base_model,
#                 GlobalAveragePooling2D(), 
#                 Dense(2, activation='softmax')
#                 ])

model.summary()

loaded_model = tf.keras.models.load_model('/home/shyam/bridge_tech/mobilenetv2-tf2/model/ec2_model/model_1_15_6_128/output/')
print('loaded_model------------->',loaded_model)

# loaded_model.summary()

preds = loaded_model.predict(test_gen,
        steps=test_gen.samples,
        max_queue_size=10, workers=1,
        use_multiprocessing=False,
        verbose=0
        )
eval = loaded_model.evaluate(test_gen,
        steps=test_gen.samples,
        max_queue_size=10, workers=1,
        use_multiprocessing=False,
        verbose=0
        )

preds = np.argmax(preds, axis=1)
pd.DataFrame(preds).to_csv('preds_128.csv')
pd.DataFrame(eval).to_csv('eval_128.csv')

print(preds,'---------',eval)  #   generator, steps=None, callbacks=None, 

print(confusion_matrix(test_gen.classes, preds))
print('Classification Report')
# target_names = ['Cats', 'Dogs', 'Horse']
print(classification_report(test_gen.classes, preds)) #, target_names=target_names))