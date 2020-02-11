#!/usr/bin/env python

from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
class DataGeneratorFactory:
  def __init__(self, input_shape,batch_size=32, num_channels=1):
    self.batch_size = batch_size
    self.num_channels = num_channels
    self.input_shape=input_shape
    self.get_generators()

  def get_generators(self):
    self.test_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True)

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

    # setting for test data
    #self.params['batch_size'] = 1
    #self.params['shuffle'] = False
    self.new_params = {
        'batch_size': 1,
        'shuffle': False
    }
    if self.num_channels == 1:
      params['color_mode'] = 'grayscale'

  def pred_gen(self,image):
      #TODO ここでカメラからの入力画像を受け付ける　imreadの部分
      #image = cv2.imread(os.environ['HOME']+"/DATA/NicoTechDice/DiceDataset/train/1/00239.png",cv2.IMREAD_GRAYSCALE)
      image = cv2.resize(image,(128, 128))
      print(image)
      print(image.shape)
      image=np.reshape(image,(1,image.shape[0],image.shape[1],1))
      return self.test_datagen.flow(image,**self.new_params)