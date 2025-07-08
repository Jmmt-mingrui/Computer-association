import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path, sheet_name):
    """加载数据表"""
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0, header=0)
    print(f"读取工作表 '{sheet_name}' 的数据：")
    print(df)
    return df

def plot_3d_bar(df, sheet_name):
    # 数据准备
    years = df.index.astype(str)  # 年份转换为字符串
    projects = df.columns.tolist()
    medal_counts = df.values

    # 创建三维画布
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 生成坐标网格
    x_pos, y_pos = np.meshgrid(np.arange(len(projects)), np.arange(len(years)))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)

    # 柱体参数
    dx = 0.35  # 柱子宽度
    dy = 0.25  # 柱子深度
    dz = medal_counts.flatten()

    # 颜色定义
    base_color = '#87CEEB'  # 天蓝色
    highlight_color = '#FF0000'  # 红色

    # --- 生成颜色数组 ---
    highlight_years = {'1900', '2024', '1920'}  # 需要标红的年份
    colors = []
    for x_idx, y_idx in zip(x_pos, y_pos):
        year = years[y_idx]  # y轴对应年份
        # 判断是否标红：年份在 highlight_years 中
        if year in highlight_years:
            colors.append(highlight_color)
        else:
            colors.append(base_color)

    # --- 绘制所有柱子 ---
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz,
             color=colors, alpha=0.9, edgecolor='grey', linewidth=0.3)

    # --- 坐标轴美化 ---
    # x轴：项目名称（间隔显示，避免重叠）
    step = max(1, len(projects) // 15)  # 每15个项目显示一个标签
    ax.set_xticks(np.arange(0, len(projects), step))
    ax.set_xticklabels(projects[::step], rotation=90, ha='center', fontsize=8)

    # y轴：年份
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels(years, rotation=90, fontsize=10, va='center')

    # z轴：奖牌数
    ax.set_zlabel('数量', fontsize=12, labelpad=20)
    ax.zaxis.set_tick_params(labelsize=10)

    # --- 视角优化 ---
    ax.view_init(elev=25, azim=-45)  # 调整视角
    ax.dist = 12  # 增加视距

    # 添加标题和网格
    ax.set_title("", fontsize=18, pad=25)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "./问题3.xlsx"
    sheet_name = "法国"  # 指定工作表名称
    df = load_data(file_path, sheet_name)
    plot_3d_bar(df, sheet_name)