#导入库
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams["axes.unicode_minus"]=False
import seaborn as sns
sns.set(font= "Kaiti",style = "ticks", font_scale = 1.4)

x= np.array([56,32,78,160,240,89,91,69])
y= np.array([90,65,125,272,312,147,159,109])

#数据的导入与处理，并进行数据探索
X= x.reshape(-1,1)                   ##reshape(-1,m) 即列数确定 排列为m行数组
Y= y.reshape(-1,1)                   ##reshape(m,-1)即行数确定，排列为m列数据
plt.figure(figsize=(10,6))              #初始化窗口

plt.scatter(X,Y,s=100)                   #原数据的图  s = size 点的大小
plt.title("原始数据图")
#plt.show()

##训练模型和预测
model = LinearRegression()
model.fit(X,Y)

#x1=np.array([40,]).reshape(-1,1)        #带预测数据
#print(x1)

x1=[np.array(i) for i in X]
x1_pre=model.predict(np.array(x1))      #预测面积为40m²的房价


#数据可视化，将预测的点打印图上
plt.figure(figsize=(10,8))
plt.scatter(X,Y)  #原数据的图    #scatter散# 点

b=model.intercept_  #截距
a=model.coef_   #斜率
y = a*X+b    #原始数据按照训练好的模型画出直线
plt.plot(X,y)

y1=a*x1+b
plt.scatter(x1,y1,color="b")
plt.show()



