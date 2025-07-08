#导入相关可视化模块
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#图像显示中文问题
import matplotlib
matplotlib.rcParams["axes.unicode_minus"]=False
#设置图像可视化时所需要的主题
import seaborn as sns
sns.set(font = "Kaiti", style = "ticks", font_scale = 1.4)#kaiti 是文字样式    没有的话 没有文字且会出现报错
##数据准备
from skimage.io import imread                   ##从skimage库中引入读取图片的函数
##从skimage库引入将RGB图片转化为灰度图像的函数
from skimage.color import rgb2gray
im=imread("./图片")
imgray=rgb2gray(im)
##可视化图片
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)          #初始图像
plt.imshow(im)
plt.axis("off")                 ##不显示坐标轴
plt.title("RGB Image")

plt.subplot(1,2,2)          ##灰度图像
plt.imshow(imgray,cmap=plt.cm.gray)
plt.axis("off")
plt.title("Gray Image")
plt.show()



##素描的实现 CSDN





