'''
##输出高清图像
%config InlineBackend.figure_format="retina"
%matplotlib inline
'''
##图像显示中文问题
import matplotlib
matplotlib.rcParams["axes.unicode_minus"]=False
import seaborn as sns
sns.set(font="Kaiti", style = "ticks", font_scale=1.4)
##导入需要的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno    ##数据异常值的可视化和处理
import altair as alt
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency                            #scipy数据科学计算库
import plotly.express as ps
from pandas.plotting import parallel_coordinates
from wordcloud import WordCloud         ####       ##常用于可视化词云图
import networkx as nx               ##图的分析与可视化
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial import distance
from sklearn.experimental import enable_iterative_imputer               #sklearn 机器学习常用库 提供机器学习算法
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
#from missingpy import MissForest                        #missingpy 提供处理缺失值的相关算法

##忽略提醒

import warnings
warnings.filterwarnings("ignore")

#读取数据
data = pd.read_excel(".\指标数据2.xlsx")
##判断每个变量中是否存在缺失值
pd.isna(data).sum()

#使用可视化方法查看缺失值在数据中的分布
msno.matrix(data,figsize=(14,7),width_ratios=(13,2),color=(0.5,0.5,0.5))
plt.show()

##剔除带有缺失值的行
data1=data.dropna(axis=0)
data1.info()
##剔除带有缺失值的列
data2=data1.dropna(axis=1)
data2.info()

##可视化剔除缺失值所在的行后变量的数据分布图
plt.figure(figsize=(30,6))
plt.scatter(data2.小分类编码 ,data2.小分类名称,c="red")
plt.grid()
plt.xlabel("小分类编码")
plt.ylabel("小分类名称")
plt.title("剔除缺失值后的行")
plt.show()


##找到缺失值所在的位置
shortage_index =pd.isna(data.小分类编码)|pd.isna(data.小分类名称)
##使用缺失值前面的值进行填充
data3=data.fillna(axis=0,method="ffill")
##可视化填充后的结果
plt.figure(figsize=(10,6))
plt.scatter(data3.小分类编码[~shortage_index], data3.小分类名称[~shortage_index], c="b",marker="o",label="非缺失值")
plt.scatter(data3.小分类编码[shortage_index], data3.小分类名称[shortage_index], c="r", marker="s",label="缺失值")
plt.grid()
plt.legend(loc="upper right", fontsize = 12)
plt.xlabel("小分类编码")
plt.ylabel("小分类名称")
plt.title("使用缺失值前面的值进行填充")
plt.show()








