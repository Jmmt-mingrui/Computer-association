# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.preprocessing import MinMaxScaler


# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a

# 生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)
    return np.array(a)

# 函数sigmoid(),两个函数都可以作为激活函数（隐藏层继续使用）
def sigmoid(x):
    return (1 - np.exp(-1 * x)) / (1 + np.exp(-1 * x))

# 函数sigmoid的派生函数（隐藏层反向传播继续使用）
def derived_sigmoid(x):
    return 1 - (np.tanh(x)) ** 2

# 构造三层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层，输出层的节点数
        self.num_in = num_in + 1  # 增加一个偏置结点
        self.num_hidden = num_hidden + 1  # 增加一个偏置结点
        self.num_out = num_out

        # 激活神经网络的所有节点（向量）
        self.active_in = np.array([-1.0] * self.num_in)
        self.active_hidden = np.array([-1.0] * self.num_hidden)
        self.active_out = np.array([1.0] * self.num_out)

        # 创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden)
        self.wight_out = makematrix(self.num_hidden, self.num_out)

        # 对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.wight_in[i][j] = random_number(0.1, 0.1)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.wight_out[i][j] = random_number(0.1, 0.1)

        # 偏差（保持不变）
        for j in range(self.num_hidden):
            self.wight_in[0][j] = 0.1
        for j in range(self.num_out):
            self.wight_in[0][j] = 0.1

        # 最后建立动量因子（矩阵）
        self.ci = makematrix(self.num_in, self.num_hidden)
        self.co = makematrix(self.num_hidden, self.num_out)

    # 信号正向传播
    def update(self, inputs):
        # 数据输入输入层
        self.active_in[1:self.num_in] = inputs

        # 数据在隐藏层的处理（保持不变）
        self.sum_hidden = np.dot(self.wight_in.T, self.active_in.reshape(-1, 1))
        self.active_hidden = sigmoid(self.sum_hidden)
        self.active_hidden[0] = -1

        # 输出层改为线性激活函数（无激活函数）
        self.sum_out = np.dot(self.wight_out.T, self.active_hidden)
        self.active_out = self.sum_out  # 直接输出线性值
        return self.active_out

    # 误差反向传播
    def errorbackpropagate(self, targets, lr, m):
        if self.num_out == 1:
            targets = np.array([targets]).reshape(-1, 1)
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')

        # 误差
        error = (1 / 2) * np.dot((targets - self.active_out).T, (targets - self.active_out))

        # 输出误差信号（线性激活的导数为 1，无需乘导数）
        self.error_out = targets - self.active_out  # 直接使用误差

        # 隐层误差信号（保持不变）
        self.error_hidden = np.dot(self.wight_out, self.error_out) * derived_sigmoid(self.sum_hidden)

        # 更新权值（保持不变）
        self.wight_out = self.wight_out + lr * np.dot(self.error_out, self.active_hidden.reshape(1, -1)).T + m * self.co
        self.co = lr * np.dot(self.error_out, self.active_hidden.reshape(1, -1)).T

        self.wight_in = self.wight_in + lr * np.dot(self.error_hidden, self.active_in.reshape(1, -1)).T + m * self.ci
        self.ci = lr * np.dot(self.error_hidden, self.active_in.reshape(1, -1)).T

        return error

    # 训练
    def train(self, pattern, itera=100, lr=0.2, m=0.1):
        for i in range(itera):
            error = 0.0
            for j in pattern:
                inputs = j[0:self.num_in - 1]  # 提取输入数据（12个自变量）
                targets = j[self.num_in - 1]   # 提取目标值（1个因变量）
                self.update(inputs)
                error = error + self.errorbackpropagate(targets, lr, m)
            if i % 10 == 0:
                print('########################误差 %-.5f######################第%d次迭代' % (error, i))

    # 预测
    def predict(self, X):
        # 预测结果
        self.update(X)
        return self.active_out


# 读取数据
data1 = pd.read_excel('美国processed_data.xlsx')  # 假设数据存储在data.xlsx文件中
data2 = pd.read_excel('美国processed_data.xlsx')  # 假设数据存储在data.xlsx文件中

# 2. 提取需要归一化的列（排除第一列）
columns_to_normalize = data1.columns[1:]  # 从第二列到最后一列
X = data1[columns_to_normalize].values

# 3. 初始化归一化器并拟合数据
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)  # 对 X 进行归一化

# 4. 将归一化后的数据替换回原表格
data_normalized = data1.copy()  # 复制原始数据
data_normalized[columns_to_normalize] = X_normalized



# 提取年份、自变量和因变量
years = data_normalized.iloc[0:33, 0].values  # 第一列是年份
X = data_normalized.iloc[0:33, 1:].values  # 前五列是自变量


y = data2.iloc[0:33, -1].values.reshape(-1,1)  # 最后一列是因变量
y = y.astype(np.float64)

print(y)
# 将数据合并为训练集
patterns = np.hstack((X, y))


# 创建BP神经网络实例
nn = BPNN(num_in=12, num_hidden=2, num_out=1)

# 训练神经网络
nn.train(patterns, itera=1000, lr=0.1, m=0.1)


# 提取第 n 行数据（例如第 0 行）
n = 33
row_data = data_normalized.iloc[n, 1:13]  # 跳过第一列，提取第 2 到第 13 列

# 将数据转换为列表
x_values = row_data.tolist()

# 将数据赋值给变量
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = x_values

# 打印结果
print(f"x1: {x1}, x2: {x2}, x3: {x3}, x4: {x4}, x5: {x5}, "
      f"x6: {x6}, x7: {x7}, x8: {x8}, x9: {x9}, x10: {x10}, "
      f"x11: {x11}, x12: {x12}")

# 预测2028年的因变量
# 假设2028年的自变量为 [x1, x2, x3, x4, x5]
#X_2028 = np.array(["1", "2(g)","2(s)","2(b)","2(n)","3(g)","3(s)","3(b)","3(n)","g","s","b"])  # 替换为实际的2028年自变量值
X_2024 = row_data.to_numpy()

prediction = nn.predict(X_2024)
print('2028年因变量的预测值为:', prediction)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(years, y, marker='o', label='Historical data')  # 绘制历史数据
plt.plot(2028, prediction, marker='*', markersize=10, color='red', label='Forecast for 2024')  # 绘制预测值
plt.xlabel('Year')
plt.ylabel('Number of medals')
plt.title('Medal values vary with year')
plt.legend()
plt.grid(True)
plt.show()