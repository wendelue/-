#-*- coding:utf-8 -*- 
#author:wenzhu


class Config(object):
    """
    """
    dic = {
           'rand_h': True,
           #'rand_h': False,
           'rand_h_num': 4,
           'height': 3,
           'pooling_stride': 1
          }  #读取数据参数设置

    input_shape = {
            'conv3d': (int(15 / dic['pooling_stride']), 101, 101, dic['rand_h_num']),
            'conv2d': (101,101,dic['rand_h_num'])
          }

    
    data_dir = '../dataset'

    #is_conv2d = False
    is_conv2d = True

    loss = 'mean_squared_error'
    optimizer = 'adam'
    
    batch = 10
    pred_batch = 100
    
    EPOCHS = 50  #迭代次数
    STEPS_PER_EPOCH = 800 #每轮迭代多少个批次
    VALIDATION_STEPS = 200



if __name__=='__main__':
    config=Config()
    print(config.STEPS_PER_EPOCH)
