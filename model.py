#-*- coding:utf-8 -*-
#author:wenzhu

import datetime
import logging
import multiprocessing
import os
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.callbacks import History
from keras.layers import (LSTM, AveragePooling3D, BatchNormalization, Conv2D,
                          Conv3D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          MaxPooling3D, Reshape)
from keras.models import Model

from read_data import dataset
import mynetworks
from Config import Config
from utils import save_data, plot_fig


from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))



def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)


class models(object):
    def __init__(self, mode, model_dir):
        self.name = 'model'
        self.mode = mode
        self.model_dir = model_dir
        self.set_log_dir()
        self.STEPS_PER_EPOCH = Config.STEPS_PER_EPOCH
        self.VALIDATION_STEPS = Config.VALIDATION_STEPS

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
               model directory.
        Returns:
            he path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]

        key = self.name
        dir_names = list(filter(lambda f: f.startswith(key), dir_names))
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        if self.mode=='training':
            dir_name = os.path.join(self.model_dir, dir_names[-2])
            os.rmdir(os.path.join(self.model_dir, dir_names[-1]))
        else:
            dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = list(filter(lambda f: f.startswith("model"), checkpoints))
        checkpoints = sorted(checkpoints)
        print(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self,model,filepath,by_name=False,exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)


    def set_log_dir(self,model_path=None):
        """Set the model log directory and epoch counter.
        model_path:If None ,or a format different form what this code uses then set a new 
        log directory and start epochs from 0. Otherwise,extract  the log directory and 
        the epoch counter form the file name.
        """
        if self.mode=='training':
            self.epoch=0
            now=datetime.datetime.now()
            #if we hanbe a model path with date and epochs use them
            if model_path:
                # Continue form we left of .Get epoch and date form the file name
                # A sample model path might look like:
                #/path/to/logs/coco2017.../DeFCN_0001.h5
                regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/DeFCN\_[\w-]+(\d{4})\.h5"
                m = re.match(regex,model_path)
                if m:
                    now=datetime.datetime(int(m.group(1)),int(m.group(2)),int(m.group(3)),
                                          int(m.group(4)),int(m.group(5)))
                    # Epoch number in file is 1-based, and in Keras code it's 0-based.
                    # So, adjust for that then increment by one to start from the next epoch
                    self.epoch = int(m.group(6)) - 1 + 1
                    print('Re-starting from epoch %d' % self.epoch)

                    # Directory for training logs
            self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.name, now))
                # Create log_dir if not exists
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

                # Path to save after each epoch. Include placeholders that get filled by Keras.
            self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(self.name))
            self.checkpoint_path = self.checkpoint_path.replace(
                    "*epoch*", "{epoch:04d}")


    def train(self,model,train_gen,val_gen,net_name,eps,is_graph=True):
        '''
        训练数据
        '''
        assert self.mode == "training", "Create model in training mode."
        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        log("\nStarting at epoch {}.\n".format(self.epoch))
        log("Checkpoint Path: {}".format(self.checkpoint_path))

        model.compile(loss=Config.loss, optimizer=Config.optimizer)

        # if os.name is 'nt':
        #     workers = 0
        # else:
        #     workers = multiprocessing.cpu_count()
        history = History()
        history = model.fit_generator(
                    train_gen,
                    initial_epoch=self.epoch,
                    epochs= eps,
                    steps_per_epoch=self.STEPS_PER_EPOCH,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    validation_steps=self.VALIDATION_STEPS,
                    #max_queue_size=100,
                    #workers=0,
                    #use_multiprocessing=False,
                )

        #model.save('save/train_model.h5')
        save_data(file_dir='saved_results/'+net_name+'/', filename='val_loss', data=history.history["val_loss"])

        save_data(file_dir='saved_results/'+net_name+'/', filename='train_loss', data=history.history["loss"])

        if is_graph:
            plot_fig(history.history["loss"], history.history["val_loss"], 'train_loss', 'val_loss')
            # fig, ax1 = plt.subplots(1,1)
            # ax1.plot(history.history["val_loss"])
            # ax1.plot(history.history["loss"])
            # plt.show()

    def pred(self, model, val_data):
        '''
        predict
        '''
        assert self.mode == "predict", "Create model in predict mode."
        self.load_weights(model,self.find_last(), by_name=True)
        results = model.predict(val_data)
        
        return results


if __name__ == "__main__":
    dataset = dataset(Config.data_dir)
    dic = Config.dic
    #训练数据生成器
    train_img_generator = dataset.data_generateor('training', rand_h=dic['rand_h'], rand_h_num=dic['rand_h_num'], height=dic['height'], pooling_stride=dic['pooling_stride'])

    #验证数据生成器
    val_img_generator = dataset.data_generateor('val', rand_h=dic['rand_h'], rand_h_num=dic['rand_h_num'], height=dic['height'], pooling_stride=dic['pooling_stride'])

    #network
    shape=(int(15/dic['pooling_stride']),101,101,1)
    network = mynetworks.model_conv3d(input_shape=shape)

    #train
    models = models(mode='training', model_dir=Config.model_dir)
    print(network.summary())
    models.train(network, train_img_generator, val_img_generator,EPOCHS=Config.EPOCHS,is_graph=True)
