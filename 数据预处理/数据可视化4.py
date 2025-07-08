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

#在可视化时将窗口切分为几个子窗口，分别绘制不同的图像
X = np.linspace(1,15)
Y = np.sin(X)

##准备数据
theta = np.linspace(-4*np.pi,4*np.pi,150)               #角度
z = np.linspace(-2,2,150)                         #z坐标                  ##数据的密集程度 尤其散点图
r = z**2+1
x=r*np.sin(theta)
y=r*np.cos(theta)
##在子图中绘制三维的图像
fig=plt.figure(figsize=(15,6))
##将坐标系设置成3d坐标系
ax1 = fig.add_subplot(121,projection="3d")          #子图1
ax1.plot(x,y,"b-")                  #绘制蓝色三维曲线图
ax1.view_init(elev=20, azim = 15)               #设置方程角和高度
ax1.set_title("3D曲线图")


ax2=plt.subplot(122,projection="3d")            #绘制子图2
ax2.scatter(x,y,z,c="r",s=20)                         #绘制红色三维散点图
ax2.view_init(elev=20, azim = 25)                     #设置轴的方位角和高程
ax2.set_title("3d散点图")
plt.subplots_adjust(wspace=0.1)
plt.show()







