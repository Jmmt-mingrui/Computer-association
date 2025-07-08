import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取数据
try:
    df = pd.read_excel('结果.xlsx', sheet_name='年份出现次数')
    y = df['出现次数'].values  # 提取数据列
    years = df['年份'].values  # 提取年份列
except KeyError:
    raise ValueError("Excel文件中必须包含 '年份' 和 '出现次数' 列")
except FileNotFoundError:
    raise FileNotFoundError("文件 '结果.xlsx' 未找到，请检查路径")

# 2. 移动平均计算
n = 6  # 移动平均窗口大小
if len(y) < n:
    raise ValueError("数据长度不足，无法进行移动平均计算")

yhat1 = np.convolve(y, np.ones(n) / n, mode='valid')  # 第一次移动平均
yhat2 = np.convolve(yhat1, np.ones(n) / n, mode='valid')  # 第二次移动平均

# 3. 预测2028年
target_year = 2028
years_ahead = target_year - years[-1]  # 计算距离当前最后一年的间隔
if years_ahead <= 0:
    raise ValueError("目标年份必须大于数据中的最大年份")

a = 2 * yhat1[-1] - yhat2[-1]  # 截距项
b = 2 * (yhat1[-1] - yhat2[-1]) / (n - 1)  # 斜率项
y_pred = a + b * years_ahead  # 预测值

# 4. 计算置信区间
residuals = y[n - 1:] - yhat1  # 残差（实际值 - 预测值）
std_dev = np.std(residuals, ddof=1)  # 样本标准差
dof = len(residuals) - 1  # 自由度

# 定义置信水平
confidence_levels = [0.90, 0.95, 0.99]
results = []

for cl in confidence_levels:
    t_critical = t.ppf((1 + cl) / 2, dof)  # t分布临界值
    margin_of_error = t_critical * std_dev * np.sqrt(1 + years_ahead / len(residuals))

    lower_bound = max(y_pred - margin_of_error, 0)  # 下界≥0
    upper_bound = y_pred + margin_of_error

    results.append({
        "confidence_level": cl,
        "prediction": y_pred,
        "lower": lower_bound,
        "upper": upper_bound
    })
    print(f"{int(cl * 100)}% 置信区间: [{lower_bound:.2f}, {upper_bound:.2f}]")

# 5. 可视化
plt.figure(figsize=(12, 6))

# 绘制历史数据
plt.plot(years, y, 'bo-', label='历史数据')

# 绘制移动平均
plt.plot(years[n - 1: n - 1 + len(yhat1)], yhat1, 'r--', label='第一次移动平均')
plt.plot(years[2 * n - 2: 2 * n - 2 + len(yhat2)], yhat2, 'g--', label='第二次移动平均')

# 绘制预测点（2028年）
pred_x = years[-1] + years_ahead
plt.plot(pred_x, y_pred, 'm*', markersize=15, label=f'{target_year}年预测值')

# 绘制置信区间（以95%为例）
cl_95 = next(item for item in results if item["confidence_level"] == 0.95)
plt.errorbar(pred_x, cl_95["prediction"],
             yerr=[[cl_95["prediction"] - cl_95["lower"]], [cl_95["upper"] - cl_95["prediction"]]],
             fmt='none', ecolor='purple', capsize=10, label='95% 置信区间')

plt.axvline(x=years[-1], color='gray', linestyle=':', label='当前年份')
plt.title(f'年份出现次数预测（目标年份：{target_year}）')
plt.xlabel('年份')
plt.ylabel('出现次数')
plt.legend()
plt.grid(True)
plt.show()