#!/usr/bin/python3
#-*- coding:utf-8 -*-
#author:wenzhu

import os
import random
import pickle

class dataset(object):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.image_path=[]
        self.batch = 10
        
    def read(self, file_name):
        file = open(file_name, "rb")
        data=pickle.load(file)
        file.close()
        return data
        
    def load_info(self, cate):
        try:
            assert cate in ['training', 'val']
        except AssertionError as ae:
            print("参数'cate'的值不正确，请输入'training'或者'val'.")

        images = os.listdir(self.data_dir)
        random.seed(0)
        random.shuffle(images)

        img_num=len(images)
        if (cate == 'training'):
            images = images[: int(0.8 * img_num)]
        else:
            images = images[int(0.8 * img_num) :]
            
        for img in images:
            img_path = os.path.join(self.data_dir, img)
            self.image_path.append(img_path)
            
    def data_generateor(self, cate):
        
        self.load_info(cate)

        while True:
            try:
                batch_img = []
                random_path = random.sample(self.image_path,self.batch)
                for i in range(self.batch):
                    path = random_path[i]
                    data = self.read(path)
                    batch_img.append(data)
                yield batch_img
            except StopIteration:
                break
            except (GeneratorExit, KeyboardInterrupt):
                raise
            

if __name__ == '__main__':
    dataset = dataset('../dataset')
    batch_img = dataset.data_generateor('training')
    print(next(batch_img)[0])
