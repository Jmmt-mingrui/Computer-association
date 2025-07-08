import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# 固定随机种子
SEED = 16
np.random.seed(SEED)

# 设置绘图参数
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 数据加载
data1 = pd.read_excel("./食用.xlsx")  # 已知数据
data2 = pd.read_excel("./食用.xlsx")  # 要进行预测的数据

# 提取最后11列作为特征
data2 = data2.iloc[:, -6:-1]
data11 = data1.iloc[:, -6:-1]

# 确保列名为字符串
data11.columns = data11.columns.astype(str)
data2.columns = data2.columns.astype(str)

# 数据集分割
x_train = data11
y_train = data1.iloc[:, -1]  # 假设最后一列是目标变量
print(y_train)
train_x, test_x, train_y, test_y = train_test_split(
    x_train, y_train,
    test_size=0.3,
    random_state=SEED
)

# 网格搜索调参
param_grid = {
    'n_estimators': [5, 10, 20, 50, 100, 200],  # 决策树的数量
    'max_depth': [3, 5, 7, 15],  # 最大树深
    'max_features': [0.6, 0.7, 0.8, 1]  # 决策树划分时考虑的最大特征数
}

# 初始化随机森林模型，并设置 random_state
rf = RandomForestRegressor(random_state=SEED)

# 初始化 GridSearchCV，不设置 random_state
grid = GridSearchCV(rf, param_grid=param_grid, cv=3, n_jobs=-1)
grid.fit(train_x, train_y)

# 最佳模型
rf_reg = grid.best_estimator_
print("最佳模型参数：")
print(rf_reg.get_params())

# 特征重要性分析
feature_names = x_train.columns
feature_importances = rf_reg.feature_importances_
indices = np.argsort(feature_importances)

print("\n特征排序：")
for index in indices:
    print(f"特征 {feature_names[index]} 的重要性: {feature_importances[index]:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.title('The importance of different features in a random forest model')
plt.bar(range(len(feature_importances)), feature_importances[indices], color='b')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance score")
plt.tight_layout()
plt.show()

# 模型评估
score = rf_reg.score(test_x, test_y)
print(f"\n模型在测试集上的R²分数: {score:.4f}")


# 将特征重要性得分乘以表格列的数值并生成新表格
def create_weighted_table(data, feature_importances, feature_names):
    """
    将特征重要性得分乘以表格列的数值并生成新表格
    :param data: 原始数据表
    :param feature_importances: 特征重要性得分
    :param feature_names: 特征名称
    :return: 加权后的数据表
    """
    # 创建新表格
    weighted_data = data.copy()

    # 对每一列进行加权
    for i, col in enumerate(feature_names):
        weighted_data[col] = data[col] * feature_importances[i]

    return weighted_data


# 生成加权后的表格
weighted_data = create_weighted_table(data11, feature_importances, feature_names)

# 将年份列添加到新表格的第一列
weighted_data.insert(0, 'Year', data1.iloc[:, 0])  # 假设第一列是年份

# 输出加权后的表格
print("\n加权后的表格（前5行）：")
print(weighted_data.head())

# 保存加权后的表格
output_file = "韩国随机加权结果.xlsx"
weighted_data.to_excel(output_file, index=False)
print(f"\n加权后的结果已保存至 {output_file}")