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
list = {2023:49/54, 2022:36/49, 2021:39/51, 2020:22/57}
#绘制一条简单曲线
X = np.linspace(2020,2023)
Y = np.sin(X)
plt.figure(figsize=(10,6))                   #图像的大小（宽：10，高：6）
plt.plot(X,Y,"r-s")                    #绘制X,Y，红色，直线，*形
plt.xlabel("X轴")                            #X轴名字
plt.ylabel("Y轴")                            #Y轴名字
plt.title("y=sin(x)")                        #图像的名字title
plt.grid()                                   #图像中添加网格线
plt.show()              #显示图像