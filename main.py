#-*- coding:utf-8 -*-
import os

import numpy as np
import tensorflow as tf

from read_data import dataset
from model import models
import mynetworks
from Config import Config
from utils import save_data, plot_fig

from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))


def train(dataset,network,models,dic):
    #训练数据生成器
    train_img_generator = dataset.data_generateor('training', rand_h=dic['rand_h'], rand_h_num=dic['rand_h_num'], height=dic['height'], pooling_stride=dic['pooling_stride'])
    
    #验证数据生成器
    val_img_generator = dataset.data_generateor('val', rand_h=dic['rand_h'], rand_h_num=dic['rand_h_num'], height=dic['height'], pooling_stride=dic['pooling_stride'])

    #train
    #models.load_weights(models.find_last(),by_name=True)
    models.train(network, train_img_generator, val_img_generator, EPOCHS=Config.EPOCHS, is_graph=True)
    
    
def predict(dataset,network,models,dic):
    #测试数据
    #valdata = dataset.
    val_data = dataset.data_generateor('val', rand_h=dic['rand_h'], rand_h_num=dic['rand_h_num'], height=dic['height'], pooling_stride=dic['pooling_stride'], batch=Config.pred_batch)
    val_data = next(val_data)
    #print(np.shape(val_data[0]))
    pred = models.pred(network, val_data=val_data[0])

    #保存数据
    real_data = val_data[1]
    save_data(file_dir='saved_results/conv2d/', filename='predict', data=pred)
    save_data(file_dir='saved_results/conv2d/', filename='real_data', data=real_data)

    #画图
    print(np.shape(pred),np.shape(real_data))
    plot_fig(real_data[:,0],pred[:,0],'real','pred')


if __name__ == "__main__":
    dic = Config.dic
    shape = Config.input_shape['conv2d']
    network = mynetworks.model_conv2d(input_shape=shape)
    print("please input：train or pred!")
    i = input()
    if i == 'train':
        train_dataset = dataset(data_dir=Config.data_dir)
        models = models(mode="training", model_dir=Config.model_dir)
        train(train_dataset,network=network, models=models,dic=dic)
    elif i == 'pred':
        pred_dataset = dataset(data_dir=Config.data_dir)
        models = models(mode="predict", model_dir=Config.model_dir)
        predict(pred_dataset,network=network,models=models,dic=dic)
    else:
        print("输入错误，请重新输入！")

        