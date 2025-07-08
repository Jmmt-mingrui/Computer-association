import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------- 定义 load_data 函数 -------------------------
def load_data(file_path):
    """
    加载数据表，第一列是年份，后面是变量
    :param file_path: 文件路径
    :return: DataFrame
    """
    df = pd.read_excel(file_path)
    print("原始数据：")
    print(df.head())  # 仅打印前5行示例
    return df


# ------------------------- 定义 time_weighted 函数 -------------------------
def time_weighted(df):
    """
    对变量列进行时间加权（按时间顺序分配权重，而非年份数值）
    :param df: 数据表
    :return: 加权后的 DataFrame
    """
    df_weighted = df.copy()
    # 生成时间权重序列（1到31）
    time_weights = np.arange(1, len(df) + 1)
    for col in df.columns[1:]:  # 跳过年份列
        df_weighted[col] = df[col] * time_weights
    return df_weighted


# ------------------------- 定义 plot_weighted_trend 函数 -------------------------
def plot_weighted_trend(df_weighted, prediction=None):
    """
    绘制趋势图（新增预测点功能）
    :param df_weighted: 加权后的数据表
    :param prediction: 预测值字典
    """
    plt.figure(figsize=(14, 8))

    # 绘制历史数据
    for col in df_weighted.columns[1:]:
        plt.plot(df_weighted['Year'], df_weighted[col],
                 marker='o', linestyle='-', linewidth=2, label=col)

    # 绘制预测点
    if prediction:
        for col in prediction.keys():
            if col != 'Year':
                plt.scatter(2028, prediction[col],
                            color='red', marker='*', s=200, zorder=10,
                            label=f'{col}Predicted value')

    plt.title('Trend analysis of time-weighted variables', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of medals', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(list(df_weighted['Year']) + [2028], rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------- 定义 predict_2028 函数 -------------------------
def predict_2028(df_weighted):
    """预测2028年的加权变量值"""
    predictions = {'Year': 2028}
    # 使用线性回归预测
    X = np.arange(1, len(df_weighted) + 1).reshape(-1, 1)  # 时间权重1~31
    for col in df_weighted.columns[1:]:  # 遍历所有变量列
        y = df_weighted[col].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict([[32]])[0][0]  # 2028年是第32个时间权重
        predictions[col] = round(pred, 2)  # 保留两位小数
    return predictions


# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    # 1. 加载数据
    file_path = "中国随机加权结果.xlsx"  # 替换为实际文件路径
    df = load_data(file_path)  # 调用 load_data 函数

    # 2. 时间加权处理
    df_weighted = time_weighted(df)
    print("\n时间加权后的表格（前5行）：")
    print(df_weighted.head())

    # 3. 预测2028年数据
    prediction = predict_2028(df_weighted)
    print("\n2028年预测值：")
    print(prediction)

    # 4. 绘制含预测值的趋势图
    plot_weighted_trend(df_weighted, prediction)

    # 5. 保存含预测值的完整数据
    # 将 prediction 字典转换为 DataFrame
    prediction_df = pd.DataFrame([prediction])

    # 使用 pd.concat 合并数据
    df_full = pd.concat([df_weighted, prediction_df], ignore_index=True)

    output_file = "中国时间加权结果_含预测.xlsx"
    df_full.to_excel(output_file, index=False)
    print(f"\n完整结果已保存至 {output_file}")