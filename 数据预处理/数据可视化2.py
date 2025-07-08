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
plt.figure(figsize=(15,12))
plt.subplot(2,2,1)  #4个窗口中第一个子窗口
plt.plot(X, Y, "b-.s")         #-.是虚线    s 是矩形方块
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")
plt.title(r"y=$\sum sin(x)$")

#histdata = np.array([1,1,2,3,4,5,4,1,6,7,8])
plt.subplot(2,2,2)  #四个窗口中的第二个子窗口
histdata = np.random.randn(200,1)   #生成数据  服从正态分布的随机数 数组
plt.hist(histdata,10)           #生成10个柱状图
plt.xlabel("取值")                #X坐标轴的label，中文
plt.ylabel("频数")                #Y坐标轴的label，中文
plt.title("直方图")                #图像的名字title，中文


plt.subplot(2,1,2)
plt.step(X,Y, c= "r", linewidth=3)              #step阶梯图，红色，线宽3，添加标签label
plt.plot(X,Y,"o--",color = "grey", alpha = 0.5,label= "sin(x)")             #o--曲线 添加灰色曲线
plt.xlabel("X")
plt.ylabel("Y")
plt.title("bar")
plt.legend(loc = "lower right", fontsize = 16 )   #图例在左下角，字体大小为16  图例紧跟label
xtick = [0,5,10,15]
xticklabel = [str(x) + "辆" for x in xtick]
plt.xticks(xtick,xticklabel,rotation=90)#x轴的坐标取值，倾斜角为45度   位置 坐标名称 旋转角度逆时针旋转
plt.subplots_adjust(hspace=0.35)        #调整子图之间的水平空间距离
plt.show()
