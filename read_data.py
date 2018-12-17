#-*- coding:utf-8 -*-
#author:wenzhu

import os
import random
import pickle
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image as Image
from Config import Config

class dataset(object):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.train_img_path = []
        self.val_img_path = []
        self.conv2d = Config.is_conv2d


    def read(self, file_name):
        file = open(file_name, "rb")
        data=pickle.load(file)
        file.close()
        return data


    def pooling(self,data, stride, batch):
        '''
        对15帧时序雷达图平均间隔采样
        stride：采样步长
        '''
        l = len(data[0, 0, 0, 0, :])
        pooling_imgs = np.ones((batch, int(15 / stride), 101, 101, l))
        for i in range(0,15,stride):
            pooling_imgs[:, int(i / stride), :, :,:] = data[:, i, :, :,:]
            #pooling_imgs[:, i, :, :] = data[:, int(i*stride), :, :]

        if self.conv2d:
            pooling_imgs =np.reshape(pooling_imgs[:,-1,:,:,:],(-1,101,101,l))

        del data

        return pooling_imgs


    def random_height(self,data, num):
        '''
        data=[batch_img_0,batch_img_1,batch_img_2,batch_img_3]
        num：随机选取的高度个数，默认为1
        '''
        random_h_imgs = random.sample(data, num)
        imgs_h = np.concatenate(random_h_imgs, 4)
        del random_h_imgs
        
        return imgs_h


    def data_sep(self,data,batch):
        '''
        输入读取的数据，将四个高度的云图分离
        '''
        batch_img_0 = np.ones((batch, 15, 101, 101,1))
        batch_img_1 = np.ones((batch, 15, 101, 101,1))
        batch_img_2 = np.ones((batch, 15, 101, 101,1))
        batch_img_3 = np.ones((batch, 15, 101, 101,1))
        label_img = np.ones((batch,1))
        id_img = []

        for i in range(batch):
            info = data[i]
            batch_img = np.reshape(info[1], [15, 4, 101, 101,1])
            batch_img_0[i] = batch_img[:, 0, :, :,:]
            batch_img_1[i] = batch_img[:, 1, :, :,:]
            batch_img_2[i] = batch_img[:, 2, :, :,:]
            batch_img_3[i] = batch_img[:, 3, :, :,:]
            label_img[i] = info[2]
            id_img.append(info[0])

        del data, batch_img, info
        
        return batch_img_0, batch_img_1, batch_img_2, batch_img_3, label_img, id_img


    def load_info(self, cate):
        try:
            assert cate in ['training', 'val']
        except AssertionError as ae:
            print("参数'cate'的值不正确，请输入'training'或者'val'.")

        images = os.listdir(self.data_dir)

        images.sort()
        random.seed(8)
        random.shuffle(images)

        img_num=len(images)
        if (cate == 'training'):
            images = images[: int(0.8 * img_num)]
            random.shuffle(images)
            for img in images:
                img_path = os.path.join(self.data_dir, img)
                self.train_img_path.append(img_path)
        else:
            images = images[int(0.8 * img_num) :]
            random.shuffle(images)
            for img in images:
                img_path = os.path.join(self.data_dir, img)
                self.val_img_path.append(img_path)


    def data_generateor(self, cate, rand_h=False, rand_h_num=1, height=3,pooling_stride=1,batch=Config.batch):
        '''
        cate: 训练数据和测试数据选择
        rand_h: 是否随机选择四个高度雷达图
        rand_h_num: rand_h为真，随机选择不同高度雷达图的数量
        height: 可以输入[0,1,2,3]中的数值，选择高度
        pooling_stride: 对15帧雷达图平均间隔采样，默认步长为1，即保留全部帧
        '''
        
        self.load_info(cate)
        bt = 0
        bv = 0

        while True:
            try:
                batch_img = []
                if cate == 'training':
                    batch_path = self.train_img_path[bt * batch : (bt + 1) * batch]
                    bt += 1
                    if bt >= len(self.train_img_path) // batch:
                        bt = 0
                else:
                    batch_path = self.val_img_path[bv * batch : (bv + 1) * batch]
                    bv += 1
                    if bv >= len(self.val_img_path) // batch:
                        bv = 0
                for i in range(batch):
                    path = batch_path[i]
                    data = self.read(path)
                    batch_img.append(data)

                batch_img_0, batch_img_1, batch_img_2, batch_img_3, label_img, id_img = self.data_sep(batch_img,batch=batch)

                if rand_h:
                    imgs_h = self.random_height([batch_img_0, batch_img_1, batch_img_2, batch_img_3], rand_h_num)
                    imgs_h = self.pooling(imgs_h,pooling_stride,batch=batch)
                    #print('random selected!')
                    yield (imgs_h, label_img)
                else:
                    if height == 0:
                        batch_img_0 = self.pooling(batch_img_0, pooling_stride,batch=batch)
                        #print('batch_img_0 selected!')
                        yield (batch_img_0, label_img)
                    elif height == 1:
                        batch_img_1 = self.pooling(batch_img_1, pooling_stride,batch=batch)
                        #print('batch_img_1 selected!')
                        yield (batch_img_1, label_img)
                    elif height == 2:
                        batch_img_2 = self.pooling(batch_img_2, pooling_stride,batch=batch)
                        #print('batch_img_2 selected!')
                        yield (batch_img_2, label_img)
                    elif height == 3:
                        batch_img_3 = self.pooling(batch_img_3, pooling_stride,batch=batch)
                        #print('batch_img_2 selected!')
                        yield (batch_img_3, label_img)
            except StopIteration:
                break
            except (GeneratorExit, KeyboardInterrupt):
                raise
            



if __name__ == '__main__':
    train_dataset = dataset('../dataset')
    batch_training_generator = train_dataset.data_generateor('training', rand_h=True, rand_h_num=4, height=3, pooling_stride=3)
    batch_val_generator = train_dataset.data_generateor('val', rand_h=True, rand_h_num=4, height=3, pooling_stride=3, batch=10)
    for i in range(900):
        batch_training = next(batch_training_generator)
        batch_val = next(batch_val_generator)
        print(np.shape(batch_training[0]),np.shape(batch_val[0]))
    #print(batch_training[1], batch_val[1])



        #展示某一高度的15张云图
    # j = 9
    # for i in range(15):
    #     image = Image.fromarray(batch_img_3[j,i,:,:].astype(np.uint8))
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.show()
    #     image.save('leida_figs/leida_figs4/'+id_img[j]+'height_3_' + str(i) + '.png')

