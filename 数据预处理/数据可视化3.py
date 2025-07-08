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


##准备需要的网格数据
x = np.linspace(-4,4,num = 50) #将x分为多少份  延y方向
y = np.linspace(-4,4,num = 50)  #将y分为多少份  延x方向
X,Y = np.meshgrid(x,y) #一维数组转化为二维矩阵 X 在y方向复制  y在X方向复制
Z = np.sin(np.sqrt(X**2+Y**2))
#绘制可视化三维曲面图
fig = plt.figure(figsize = (10,6))
##将坐标设置成3D坐标系
ax1 = fig.add_subplot(111, projection ="3d" )
##绘制曲面图，rstrid:行的跨度，cstride：列的跨度，cmap：颜色，alpha：透明度
ax1.plot_surface(X,Y,Z,rstride = 10, cstride = 1,alpha = 0.5, cmap = plt.cm.coolwarm)        #cmap = plt.cm.coolwarm 热力图越向上颜色越r   行延x方向 列延y方向
##制作Z的等高线，投影位置在z=1的平面
cset = ax1.contour(X,Y,Z,zdir = "z", offset=1,cmap = plt.cm.CMRmap)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_xlim(-4,4)
ax1.set_ylim(-4,4)
ax1.set_zlabel("Z")
ax1.set_zlim(-4,4)
ax1.set_title("曲面图和等高线")
plt.show()
