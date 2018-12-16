#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras import backend as K
from keras.callbacks import History
from keras.layers import (LSTM, AveragePooling3D, BatchNormalization, Conv2D,
                          Conv3D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          MaxPooling3D, Reshape)
from keras.models import Model


def model_conv2d(input_shape):
    # Model parameters
    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    #c1 = Convolution2D(filters=8, 7, 7, border_mode='same', init='he_uniform', activation='relu')(inpN)
    c1 = Conv2D(filters=8, kernel_size= (7,7), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(inpN)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(filters=8, kernel_size= (7,7), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    pool_1 = MaxPooling2D()(c1)
    #drop_1 = Dropout(0.25)(pool_1)
    
    c2 = Conv2D(filters=16, kernel_size= (5,5), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(pool_1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(filters=16, kernel_size= (5,5), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    pool_2 = MaxPooling2D()(c2)
    #drop_2 = Dropout(0.25)(pool_2)
    
    c3 = Conv2D(filters=32, kernel_size= (3,3), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(pool_2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=32, kernel_size= (3,3), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    pool_3 = MaxPooling2D()(c3)
    drop_3 = Dropout(0.25)(pool_3)
    
    c4 = Conv2D(filters=32, kernel_size= (3,3), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(drop_3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=32, kernel_size= (3,3), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    pool_4 = MaxPooling2D()(c4)
    drop_4 = Dropout(0.25)(pool_4)
    
    flat = Flatten()(drop_4)
    hidden_1 = Dense(1024, kernel_initializer='glorot_uniform', activation='relu')(flat)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.3)(hidden_1)
    hidden_2 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Dropout(0.3)(hidden_2)
    hidden_3 = Dense(124, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Dropout(0.3)(hidden_3)
    out = Dense(1, activation='relu')(hidden_3)

    model = Model(outputs=out, inputs=inp)
    print(model.summary())

    return model


def model_conv3d(input_shape):
    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    
    c1 = Conv3D(filters=8, kernel_size= (5,5,5), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(inpN)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(filters=8, kernel_size= (5,5,5), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    pool_1 = MaxPooling3D(pool_size=(2,3,3))(c1)
    drop_1 = Dropout(0.25)(pool_1)
    
    c2 = Conv3D(filters=16, kernel_size= (3,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(drop_1)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(filters=16, kernel_size= (3,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    pool_2 = MaxPooling3D(pool_size=(1,3,3))(c2)
    drop_1 = Dropout(0.25)(pool_2)
    
    c3 = Conv3D(filters=32, kernel_size= (3,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(drop_1)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(filters=32, kernel_size= (3,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    pool_3 = MaxPooling3D(pool_size=(1,2,2))(c3)
    drop_3 = Dropout(0.25)(pool_3)

    flat = Flatten()(drop_3)
    hidden_1 = Dense(1024, kernel_initializer='glorot_uniform', activation='relu')(flat)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.3)(hidden_1)
    hidden_2 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Dropout(0.3)(hidden_2)
    hidden_3 = Dense(124, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Dropout(0.3)(hidden_3)
    hidden_4 = Dense(8, kernel_initializer='glorot_uniform', activation='relu')(hidden_3)
    hidden_4 = BatchNormalization()(hidden_4)
    hidden_4 = Dropout(0.3)(hidden_4)
    out = Dense(1, activation='relu')(hidden_4)
    
    model = Model(outputs=out, inputs=inp)
    
    print(model.summary())

    return model



def model_conv3d_lstm(input_shape):
    # Model parameters
    input_shape = (t_frames, rows, cols, 4)

    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    pool_0 = AveragePooling3D(pool_size=(1,4,4))(inpN) # downsample slightly ???
    
    c1 = Conv3D(filters=8, kernel_size= (1,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(pool_0)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(filters=8, kernel_size= (1,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    pool_1 = MaxPooling3D(pool_size=(1,2,2))(c1)
    drop_1 = Dropout(0.4)(pool_1)
    
    c2 = Conv3D(filters=16, kernel_size= (1,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(drop_1)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(filters=16, kernel_size= (1,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    pool_2 = MaxPooling3D(pool_size=(1,2,2))(c2)
    drop_1 = Dropout(0.4)(pool_2)
    
    flat = Reshape((15,6*6*16))(drop_1)
    flat = BatchNormalization()(flat)
    flat = Dropout(0.2)(flat)
    
    lstm_1 = LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(flat)
    
    lstm_2 = LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, dropout=0.3, recurrent_dropout=0.3)(lstm_1)
    
#    lstm_3 = LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, dropout=0.3, recurrent_dropout=0.3)(lstm_2)
    
    hidden_1 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(lstm_2)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.4)(hidden_1)
    hidden_2 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Dropout(0.4)(hidden_2)
#    hidden_2a = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
#    hidden_2a = BatchNormalization()(hidden_2a)
#    hidden_2a = Dropout(0.4)(hidden_2a)
    hidden_3 = Dense(256, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Dropout(0.4)(hidden_3)
#    hidden_4 = Dense(8, kernel_initializer='glorot_uniform', activation='relu')(hidden_3)
#    hidden_4 = BatchNormalization()(hidden_4)
#    hidden_4 = Dropout(0.5)(hidden_4)
    out = Dense(1, activation='linear')(hidden_3)
    
    model = Model(outputs=out, inputs=inp)
    
    print(model.summary())

    return model




