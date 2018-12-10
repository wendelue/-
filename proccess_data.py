#!/usr/bin/python3
#-*- coding:utf-8 -*-
#author:wenzhu

import numpy as np

from read_data import dataset

def process(data):
    length = len(data)
    batch_img = np.ones((length, 15, 4, 101, 101))
    label_img = np.ones((length,1))
    for i in range(length):
        info = data[i]
        batch_img[i] = np.reshape(info[1], [15, 4, 101, 101])
        label_img[i] = info[2]
        print(info[0] + 'has processed!')
    print('end!')
    return batch_img,label_img



if __name__ == '__main__':
    data = dataset('../dataset')
    train_data = data.data_generateor('training')
    batch_img, label_img = process(next(train_data))
    print(np.shape(batch_img), label_img)
