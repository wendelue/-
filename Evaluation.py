#-*- coding:utf-8 -*-
#author: wenzhu

import scipy.io
import pandas as pd
from utils import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score


#读取数据
pred = pd.read_csv("saved_results/model_conv2d/predict.csv")
real = pd.read_csv("saved_results/model_conv2d/real_data.csv")
pred = pred.values
real = real.values

#计算指标
mae = mean_absolute_error(real, pred)
mse = mean_squared_error(real, pred)
rmse = root_mean_squared_error(real, pred)
r2 = r2_score(real,pred)


print('mae:', mae)
print('mse:', mse)
print('rmse:', rmse)
print('r2_score:',r2)
