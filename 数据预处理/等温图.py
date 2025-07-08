import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_3d_data(file_path, countries):
    """加载并处理三维数据（带数据清洗）"""
    dfs = []
    for country in countries:
        # 加载原始数据
        df = pd.read_excel(file_path, sheet_name=country, index_col=0)

        # 清洗年份索引
        year_series = df.index.to_series()
        df.index = (
            year_series
            .astype(str)
            .str.extract('(\d+)', expand=False)
            .pipe(pd.to_numeric, errors='coerce')
            .fillna(method='ffill')
            .astype(int)
        )

        # 清洗数值列
        df.iloc[:, 1:] = (
            df.iloc[:, 1:]
            .replace(' ', np.nan)
            .fillna(method='ffill')
            .apply(pd.to_numeric, errors='coerce')
        )

        # 检查缺失值
        if df.isnull().values.any():
            df = df.fillna(method='ffill').fillna(0)  # 双重填充确保无NaN

        # 取最后五年并转换格式
        df = df.tail(5).stack().reset_index()
        df.columns = ['Year', 'Variable', 'Value']
        df['Country'] = country
        dfs.append(df)

    merged_data = pd.concat(dfs)
    # 最终数据验证
    assert not merged_data.isnull().any().any(), "合并数据存在缺失值"
    return merged_data


def plot_3d_bubble(data):
    """绘制带连线的三维气泡图"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    country_colors = {
        'United States': '#FF6B6B',
        'Japan': '#4ECDC4',
        'India': '#FFE66D'
    }

    variables = data['Variable'].unique()
    var_mapping = {var: idx for idx, var in enumerate(variables)}
    print("变量映射:", var_mapping)

    for country in data['Country'].unique():
        country_data = data[data['Country'] == country]
        color = country_colors[country]

        # 按变量分组绘制时间趋势线
        for var_name in country_data['Variable'].unique():
            var_df = country_data[country_data['Variable'] == var_name]
            var_df = var_df.sort_values('Year')  # 按时间排序

            # 至少需要2个点才能画线
            if len(var_df) < 2:
                continue

            x_line = var_df['Year'].astype(int)
            y_line = var_df['Variable'].map(var_mapping)
            z_line = var_df['Value']

            ax.plot(x_line, y_line, z_line,
                    color=color,
                    linestyle='--',
                    linewidth=1,
                    alpha=0.5)  # 调低透明度避免喧宾夺主

        # 气泡绘制
        x = country_data['Year'].astype(int)
        y = country_data['Variable'].map(var_mapping)

        if y.isnull().any():
            missing = country_data.loc[y.isnull(), 'Variable'].unique()
            raise ValueError(f"未映射变量: {missing} (国家: {country})")

        z = country_data['Value']

        if z.nunique() == 0:
            size = np.full_like(z, 50, dtype=float)
        else:
            size = (z - z.min()) / (z.max() - z.min()) * 100 + 10

        size = size.astype(float)
        size = np.nan_to_num(size, nan=10.0)

        ax.scatter(x, y, z, s=size, c=color,
                   alpha=0.8, label=country, depthshade=True)

    # 坐标轴设置
    ax.set_xlabel('\nYear', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_zlabel('\nMetric values', fontsize=12)
    ax.set_yticks(list(var_mapping.values()))
    ax.set_yticklabels(var_mapping.keys())
    ax.view_init(elev=25, azim=-45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("", fontsize=16)  # 标题
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_3d_data("./等温数据.xlsx", ["United States", "Japan", "India"])
    plot_3d_bubble(data)