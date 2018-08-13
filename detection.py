
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = '00_input/train'
im_size = 100
coords_size = 28


# In[2]:


def read_csv(filename):
    from numpy import array
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res

train_gt = read_csv(train_dir+'/gt.csv')


# In[3]:


from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence

from keras.layers import (Input, concatenate, Conv2D, MaxPooling2D, 
                          UpSampling2D, Convolution2D, ZeroPadding2D, 
                          BatchNormalization, Activation, concatenate, 
                          Flatten, Dense, merge)
from keras.optimizers import rmsprop

def get_model():
    inputs = Input(shape=(im_size, im_size, 1))
    conv = Conv2D(filters=256, kernel_size=(3,3), padding='same')(inputs)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    conv = Conv2D(filters=128, kernel_size=(3,3), padding='same')(relu)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    maxpool = MaxPooling2D()(relu)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(maxpool)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(relu)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    maxpool = MaxPooling2D()(relu)
    
    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same')(maxpool)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    flatten = Flatten()(relu)
    predictions = Dense(coords_size, activation='tanh')(flatten)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model

def first_model():
    inputs = Input(shape=(im_size, im_size, 1))
    conv = Conv2D(filters=256, kernel_size=(3,3), padding='same')(inputs)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    conv = Conv2D(filters=128, kernel_size=(3,3), padding='same')(relu)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    maxpool = MaxPooling2D()(relu)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(maxpool)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(relu)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(relu)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    maxpool = MaxPooling2D()(relu)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(maxpool)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(relu)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    maxpool = MaxPooling2D()(relu)
    
    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same')(maxpool)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    flatten = Flatten()(relu)
    predictions = Dense(coords_size, activation='tanh')(flatten)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model


# In[4]:


def train_detector(
        train_gt, 
        train_img_dir, 
        fast_train=False, 
        model_func=None, 
        model_name='{epoch:d}_{val_loss:.4f}.hdf5'):
    
    def parse(train_gt, train_img_dir, info=False):
        from skimage.data import imread
        from scipy.ndimage import zoom
        train_X = np.zeros((len(train_gt), im_size, im_size, 1))
        train_Y = np.zeros((len(train_gt), coords_size))
        for i, img_name in enumerate(train_gt):
            img = imread(train_img_dir+'/'+img_name, as_grey=True)
            train_Y[i] = train_gt[img_name]
            for j in range(1, coords_size, 2):
                train_Y[i][j] *= im_size/img.shape[0]
            for j in range(0, coords_size, 2):
                train_Y[i][j] *= im_size/img.shape[1]
            train_Y[i] = (train_Y[i] / 100) 
            img = zoom(img, [im_size/img.shape[0], im_size/img.shape[1]])
            img = (img / 255) 
            train_X[i,:,:,0] = img
            del(img)
            if info and (i+1)%100 == 0:
                print('Image: ', i+1, end='\r')
        return train_X, train_Y
    
    train_X, train_Y = parse(train_gt, train_img_dir, True)
    if model_func == None:
        model = get_model()
    else:
        model = model_func()
        model_name += '_{epoch:d}_{val_loss:.4f}.hdf5'
    model.summary()
    checkpoint = ModelCheckpoint(
        model_name, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        period=1,
        save_weights_only=False)
    if fast_train:
        epochs = 1
        model.fit(train_X, train_Y, epochs=epochs, batch_size=100, validation_split=(1/6))
    else:
        epochs = 20
        try:
            model.fit(train_X, train_Y, epochs=epochs, batch_size=100, callbacks=[checkpoint], validation_split=(1/6))
            #model.save('facepoints_model.hdf5')
        except KeyboardInterrupt:
            #model.save('facepoints_model.hdf5')
            print('\nTraining interrupted')
    return model


# In[6]:


train_detector(train_gt, train_dir+'/images', False, first_model, 'first')


# In[5]:


def detect(model, test_img_dir):
    from os import listdir
    from skimage.data import imread
    from scipy.ndimage import zoom
    img_list = listdir(test_img_dir)
    data = np.zeros((len(img_list), im_size, im_size, 1))
    sizes = []
    for i, img_name in enumerate(img_list):
        img = imread(test_img_dir+'/'+img_name, as_grey=True)
        sizes.append([img_name, img.shape])
        img = zoom(img, [im_size/img.shape[0], im_size/img.shape[1]])
        img = (img / 255)
        data[i,:,:,0] = img
        del(img)
        if(i+1)%100 == 0:
            print('Image: ', i+1, end='\r')
    points = model.predict(data, 100, 1)
    ans = {}
    points *= 100
    for i in range(len(points)):
        for j in range(1, coords_size, 2):
            points[i][j] *= sizes[i][1][0]/im_size
            points[i][j] = int(points[i][j])
        for j in range(0, coords_size, 2):
            points[i][j] *= sizes[i][1][1]/im_size
            points[i][j] = int(points[i][j])
        ans[sizes[i][0]] = points[i]
    return ans

from keras.models import load_model
model = load_model('first_10_0.0039.hdf5')
ans_dict = detect(model, '00_input/test/images')


# In[6]:


ans_dict
