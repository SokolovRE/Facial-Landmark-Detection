
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = '00_input/train'
im_size = 100
coords_size = 28
fast_len = 500

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
from keras.optimizers import rmsprop

def get_model():
    inputs = Input(shape=(im_size, im_size, 1))
    conv = Conv2D(filters=16, kernel_size=(3,3), padding='same')(inputs)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same')(relu)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    maxpool = MaxPooling2D()(relu)
    
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same')(maxpool)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    maxpool = MaxPooling2D()(relu)
    
    conv = Conv2D(filters=128, kernel_size=(3,3), padding='same')(maxpool)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    maxpool = MaxPooling2D()(relu)
    
    dropout = Dropout(0.5)(maxpool)
    
    conv = Conv2D(filters=256, kernel_size=(3,3), padding='same')(dropout)
    batchnorm = BatchNormalization()(conv)
    relu = Activation('relu')(batchnorm)
    
    flatten = Flatten()(relu)
    predictions = Dense(coords_size, activation=None)(flatten)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def parse(train_img_dir, train_gt=None, fast_train=False):
    from skimage.data import imread
    from scipy.ndimage import zoom
    from os import listdir
    img_list = listdir(train_img_dir)
    if fast_train:
        data_len = fast_len
    else:
        data_len = len(img_list)
    train_X = np.zeros((data_len, im_size, im_size, 1))
    img_M = np.zeros((im_size, im_size))
    img_D = np.zeros((im_size, im_size))
    if train_gt != None:
        train_Y = np.zeros((data_len, coords_size))
        gt_M = np.zeros(coords_size)
        gt_D = np.zeros(coords_size)
    else:
        sizes = []
    for i, img_name in enumerate(img_list):
        if i == fast_len and fast_train:
            break
        img = imread(train_img_dir+'/'+img_name, as_grey=True)
        if train_gt != None:
            train_Y[i] = train_gt[img_name]
            for j in range(1, coords_size, 2):
                train_Y[i][j] *= im_size/img.shape[0]
            for j in range(0, coords_size, 2):
                train_Y[i][j] *= im_size/img.shape[1]
            gt_M += train_Y[i]
            gt_D += train_Y[i]**2
        else:
            sizes.append([img_name, img.shape])
        img = zoom(img, [im_size/img.shape[0], im_size/img.shape[1]])
        img_M += img
        img_D += img**2
        train_X[i,:,:,0] = img
        del(img)
        if (i+1)%100 == 0:
            print('Parsing image: ', i+1, end='\r')
        if train_gt != None:
            gt_M /= data_len
            gt_D /= data_len
            gt_D -= gt_M**2
            train_Y -= gt_M
            train_Y /= gt_D
        img_M /= data_len
        img_D /= data_len
        img_D -= img_M**2
        train_X[:,:,:,0] -= img_M
        train_X[:,:,:,0] /= img_D
    if train_gt != None:
        return train_X, train_Y
    else:
        return train_X, gt_M, gt_D, sizes


def train_detector(
        train_gt, 
        train_img_dir, 
        fast_train=False, 
        model_func=None, 
        model_name='{epoch:d}_{val_loss:.4f}.hdf5'):
    
    train_X, train_Y = parse(train_img_dir, train_gt, fast_train)
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
        model.fit(train_X, train_Y, epochs=epochs, batch_size=100)
    else:
        epochs = 100
        try:
            model.fit(train_X, train_Y, epochs=epochs, batch_size=100, callbacks=[checkpoint], validation_split=(1/6))
        except KeyboardInterrupt:
            print('\nTraining interrupted')
        return model


def detect(model, test_img_dir):
    from os import listdir
    from skimage.data import imread
    from scipy.ndimage import zoom
    data, gt_M, gt_D, sizes = parse(test_img_dir)
    points = model.predict(data, 100, 1)
    ans = {}
    points *= gt_D
    points += gt_M
    for i in range(len(points)):
        for j in range(1, coords_size, 2):
            points[i][j] *= sizes[i][1][0]/im_size
            points[i][j] = int(points[i][j])
        for j in range(0, coords_size, 2):
            points[i][j] *= sizes[i][1][1]/im_size
            points[i][j] = int(points[i][j])
        ans[sizes[i][0]] = points[i]
    return ans