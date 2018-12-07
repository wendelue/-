import os
import time
import random

import numpy as np


class dataset(object):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.image_info=[]
        self.batch = 10
        
    def read(self, file_name):
        f = open(file_name,"r")
        image_info = f.readline()
        f.close()
        image_label = image_info[0]
        image = image_info[1:]
        return image, image_label
        
    def load_info(self, cate):
        try:
            assert cate in ['training', 'val']
        except AssertionError as ae:
            print('参数'cate'的值不正确，请输入'training'或者'val'.')

        images = os.listdir(self.data_dir)
        random.seed(0)
        random.shuffle(images)

        img_num=len(images)
        if (category == 'training'):
            images = images[: 0.8 * img_num]
        else:
            images = images[0.8 * img_num :]
            
        for img in images:
            img_path = os.path.join(data_dir, img)
            self.image_info.append({'id': img, 'path': img_path})
            
    def data_generateor(dataset,cate):
        self.load_info(cate)

        while True:
            try:
                niubi
                haha

                pass
            except StopIteration:
                break
            
            
        
        