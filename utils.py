#-*- coding:utf-8 -*-

import scipy.io
import matplotlib.pyplot as plt

def save_data(file_dir,filename,data):
    scipy.io.savemat(file_dir + filename + '.mat', {filename: data})


def plot_fig(data1,data2,label1,label2):
    plt.plot(data1,label=label1)
    plt.plot(data2,label=label2)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    pass
