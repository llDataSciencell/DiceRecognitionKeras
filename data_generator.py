#!/usr/bin/env python

from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
class DataGeneratorFactory:
  def __init__(self, data_dir,input_shape,batch_size=32, num_channels=1):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_channels = num_channels
    self.input_shape=input_shape
    self.get_generators()

  def get_generators(self):
    train_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        zca_whitening=True)
    valid_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        zca_whitening=True)
    self.test_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        zca_whitening=True)

    params = {
        'target_size': self.input_shape,
        'classes': ['1', '2', '3', '4', '5', '6'],
        'class_mode': 'categorical',
        'batch_size': self.batch_size,
        'seed': 1234,
        'shuffle':True
    }
    if self.num_channels == 1:
      params['color_mode'] = 'grayscale'
    self.train_generator = train_datagen.flow_from_directory(
        '%s/train' % self.data_dir, **params)
    self.valid_generator = valid_datagen.flow_from_directory(
        '%s/valid' % self.data_dir, **params)
    # setting for test data
    #self.params['batch_size'] = 1
    #self.params['shuffle'] = False
    self.test_params = {
        'batch_size': 1,
        'shuffle': False,
        'target_size': self.input_shape,
        'classes': ['1', '2', '3', '4', '5', '6'],
        'class_mode': 'categorical',
        'seed': 1234,
    }
    if self.num_channels == 1:
      self.test_params['color_mode'] = 'grayscale'

    self.test_generator = self.test_datagen.flow_from_directory(
        '%s/test' % self.data_dir, **self.test_params)

    self.train_size = len(self.train_generator.filenames)
    self.valid_size = len(self.train_generator.filenames)
    self.test_size = len(self.test_generator.filenames)

  def pred_gen_using_raw_image(self,image):
      #TODO ここでカメラからの入力画像を受け付ける　imreadの部分
      #image = cv2.imread(os.environ['HOME']+"/DATA/NicoTechDice/DiceDataset/train/1/00239.png",cv2.IMREAD_GRAYSCALE)
      image = cv2.resize(image,(64, 64))
      print(image)
      print(image.shape)
      image=np.reshape(image,(1,image.shape[0],image.shape[1],1))
      return self.test_datagen.flow(image,**self.test_params)
  