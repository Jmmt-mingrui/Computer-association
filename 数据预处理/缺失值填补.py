import pandas as pd

# 读取Excel文件
skip_rows = 1
file_path = './新能源汽车平均.xlsx'

df = pd.read_excel(file_path,skiprows=skip_rows)


# 计算每列的平均值
mean_values = df.mean()

# 使用平均值填充缺失数据
df_filled = df.fillna(mean_values)

# 将处理后的数据保存到新的Excel文件
output_file_path = '新能源汽车平均1.xlsx'
df_filled.to_excel(output_file_path, index=False)
