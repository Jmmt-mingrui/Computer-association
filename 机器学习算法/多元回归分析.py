import pandas as pd
from sklearn.linear_model import LinearRegression

# 读取数据
data = pd.read_excel('./上海.xlsx')

# 提取自变量和因变量
X = data[['上海']]
y = data['造林总面积']

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 预测2025年的自然区面积
new_data = pd.DataFrame({'上海': [2025]})
predictions = model.predict(new_data)

print(predictions)
