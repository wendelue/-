#-*- coding:utf-8 -*-
#author: wenzhu

import os

import numpy as np
import tensorflow as tf

from read_data import dataset
from model import models
import mynetworks
from Config import Config
from utils import save_data, plot_fig

from keras.utils.vis_utils import plot_model
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))


def train(dataset,network,models,dic,net_name):
    #训练数据生成器
    train_img_generator = dataset.data_generateor('training', rand_h=dic['rand_h'], rand_h_num=dic['rand_h_num'], height=dic['height'], pooling_stride=dic['pooling_stride'])

    #验证数据生成器
    val_img_generator = dataset.data_generateor('val', rand_h=dic['rand_h'], rand_h_num=dic['rand_h_num'], height=dic['height'], pooling_stride=dic['pooling_stride'])

    #train
    #models.load_weights(models.find_last(),by_name=True)
    models.train(network,train_img_generator,val_img_generator,net_name=net_name,eps=Config.EPOCHS,is_graph=True)


def predict(dataset,network,models,dic,net_name):
    #测试数据
    #valdata = dataset.
    val_data = dataset.data_generateor('pred', rand_h=dic['rand_h'], rand_h_num=dic['rand_h_num'], height=dic['height'], pooling_stride=dic['pooling_stride'], batch=Config.pred_batch)
    val_data_sum = []
    pred_sum=[]
    for i in range(2000//Config.pred_batch):
        val_data_i = next(val_data)
        #print(np.shape(val_data[0]))
        pred = models.pred(network, val_data=val_data_i[0])
        val_data_sum.extend(val_data_i[1])
        pred_sum.extend(pred)

    #保存数据
    save_data(file_dir='saved_results/' + net_name+'/', filename='predict', data=pred_sum)
    save_data(file_dir='saved_results/' + net_name+'/', filename='real_data', data=val_data_sum)

    #画图
    print(np.shape(pred_sum),np.shape(val_data_sum))
    plot_fig(val_data_sum, pred_sum, 'real', 'pred')


if __name__ == "__main__":
    dic = Config.dic
    shape = Config.input_shape['conv2d']
    network,net_name =mynetworks.model_conv2d(input_shape=shape)
    plot_model(network, to_file='models/net_fig/' + net_name + '.png', show_shapes=True)
    print("Model structure has saved in 'models/net_fig/"+ net_name+ ".png'")
    print("please input：train or pred!")
    i = input()
    if i == 'train':
        train_dataset = dataset(data_dir=Config.data_dir)
        models = models(mode="training", model_dir='./models/'+net_name)

        train(train_dataset,network=network, models=models,dic=dic,net_name=net_name)
    elif i == 'pred':
        pred_dataset = dataset(data_dir=Config.data_dir)
        models = models(mode="predict", model_dir='./models/'+net_name)

        predict(pred_dataset,network=network,models=models,dic=dic,net_name=net_name)
    else:
        print("输入错误，请重新输入！")
