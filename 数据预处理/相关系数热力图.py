import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.preprocessing import MinMaxScaler

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取数据
def load_data(file_path):
    """
    加载数据表，第一列是项目类型，第一行是国家，中间是相关性系数
    :param file_path: 文件路径
    :return: DataFrame
    """
    df = pd.read_excel(file_path, index_col=0)  # 第一列作为行索引
    print("原始数据：")
    print(df)
    return df

# 2. 绘制热力图
def plot_heatmap(df):
    """
    绘制热力图，横坐标是国家，纵坐标是项目类型，中间是相关性系数
    :param df: 数据表
    """
    plt.figure(figsize=(12, 8))  # 设置画布大小

    # 使用 seaborn 绘制热力图
    sns.heatmap(
        df,  # 数据
        annot=True,  # 显示数值
        fmt=".2f",  # 数值格式（保留两位小数）
        cmap="Blues_r",  # 颜色映射
        center=0,  # 颜色中心点为 0
        linewidths=0.5,  # 单元格边框宽度
        linecolor="black",  # 单元格边框颜色
        mask=df.isnull(),  # 空值部分留白
        cbar_kws={"shrink": 0.5}  # 调整颜色条大小
    )

    # 设置图表属性
    plt.title("Correlation heat map of country to project type", fontsize=16)
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("Project type", fontsize=12)
    plt.xticks(rotation=45)  # 旋转横坐标标签
    plt.yticks(rotation=0)  # 纵坐标标签不旋转
    plt.tight_layout()  # 自动调整布局
    plt.show()

# 主程序
if __name__ == "__main__":
    # 1. 加载数据
    file_path = "相关系数.xlsx"  # 替换为实际文件路径
    df = load_data(file_path)

    # 2. 绘制热力图
    plot_heatmap(df)