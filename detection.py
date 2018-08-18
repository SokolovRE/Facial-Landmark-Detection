
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = '00_input/train'
im_size = 160
coords_size = 28

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


from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence

from keras.layers import (Input, concatenate, Conv2D, MaxPooling2D, 
                          UpSampling2D, Convolution2D, ZeroPadding2D, 
                          BatchNormalization, Activation, concatenate, 
                          Flatten, Dense, merge, Dropout)
from keras.optimizers import Adam

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
    predictions = Dense(coords_size, activation=None)(flatten)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model


def get_model():
    inputs = Input(shape=(im_size, im_size, 1))
    dropout = Dropout(0.2)(inputs)
    conv = Conv2D(filters=16, kernel_size=(3,3), padding='same')(dropout)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    
    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same')(batchnorm)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    maxpool = MaxPooling2D()(batchnorm)
    
    conv = Conv2D(filters=48, kernel_size=(3,3), padding='same')(maxpool)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    
    dropout = Dropout(0.3)(batchnorm)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(dropout)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    maxpool = MaxPooling2D()(batchnorm)
    
    conv = Conv2D(filters=96, kernel_size=(3,3), padding='same')(maxpool)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    
    conv = Conv2D(filters=128, kernel_size=(3,3), padding='same')(batchnorm)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    maxpool = MaxPooling2D()(batchnorm)
    
    dropout = Dropout(0.3)(maxpool)
    
    conv = Conv2D(filters=192, kernel_size=(3,3), padding='same')(dropout)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    
    conv = Conv2D(filters=256, kernel_size=(3,3), padding='same')(batchnorm)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    maxpool = MaxPooling2D()(batchnorm)
    
    conv = Conv2D(filters=384, kernel_size=(3,3), padding='same')(maxpool)
    relu = Activation('relu')(conv)
    batchnorm = BatchNormalization()(relu)
    
    dropout = Dropout(0.5)(batchnorm)
    
    flatten = Flatten()(dropout)
    predictions = Dense(coords_size, activation=None)(flatten)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=Adam(0.001, decay=0.00002), loss='mean_squared_error')
    return model


def train_detector(
        train_gt, 
        train_img_dir, 
        fast_train=False, 
        model_func=None, 
        model_name='{epoch:d}_{val_loss:.4f}.hdf5'):
    
    def parse(train_gt, train_img_dir, info=False, fast_train=False):
        from skimage.data import imread
        from scipy.ndimage import zoom
        if fast_train:
            train_X = np.zeros((500, im_size, im_size, 1))
            train_Y = np.zeros((500, coords_size))
        else:
            train_X = np.zeros((len(train_gt), im_size, im_size, 1))
            train_Y = np.zeros((len(train_gt), coords_size))
        for i, img_name in enumerate(train_gt):
            if i == 500 and fast_train:
                break
            img = imread(train_img_dir+'/'+img_name, as_grey=True)
            train_Y[i] = train_gt[img_name]
            for j in range(1, coords_size, 2):
                train_Y[i][j] *= im_size/img.shape[0]
            for j in range(0, coords_size, 2):
                train_Y[i][j] *= im_size/img.shape[1]
            img = zoom(img, [im_size/img.shape[0], im_size/img.shape[1]])
            img = (img / 255) 
            train_X[i,:,:,0] = img
            del(img)
            if info and (i+1)%100 == 0:
                print('Image: ', i+1, end='\r')
        return train_X, train_Y
    
    train_X, train_Y = parse(train_gt, train_img_dir, True, fast_train)
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
        model.fit(train_X, train_Y, epochs=epochs, batch_size=150)
    else:
        epochs = 100
        try:
            model.fit(train_X, train_Y, epochs=epochs, batch_size=150, callbacks=[checkpoint], validation_split=(1/6))
        except KeyboardInterrupt:
            print('\nTraining interrupted')
    return model


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
    for i in range(len(points)):
        for j in range(1, coords_size, 2):
            points[i][j] *= sizes[i][1][0]/im_size
            points[i][j] = int(points[i][j])
        for j in range(0, coords_size, 2):
            points[i][j] *= sizes[i][1][1]/im_size
            points[i][j] = int(points[i][j])
        ans[sizes[i][0]] = points[i]
    return ans