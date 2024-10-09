import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import re

# # 定义CSV文件路径
# file_paths = ['apple.csv', 'orange.csv', 'peach.csv', 'air.csv', 'watermelon_3458.csv']
# # file_paths = ['apple_2500.csv', 'orange_2500.csv', 'peach_2500.csv', 'air_2500.csv']
# # 数据集标签
# labels = ['Apple', 'Orange', 'Peach', 'Air', 'Watermelon']

# 读取当前目录下每个.csv文件
file_paths = []
for file in os.listdir('.'):
    if re.match(r'^test_data_.*\.csv$', file):
        print(file)
        file_paths.append(file)
labels = ['Air', 'Orange', 'Peach']


# 读取每个CSV文件并执行PCA，同时确保所有结果具有相同的形状
min_samples = None
pca_results = []
all_data =[]
row_len = []
for file_path in file_paths:
    # 读取CSV文件
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df.values,columns = ['C%d'%i for i in range(24)]+['label'])
    # 提取第五列开始的24个传感器数据列
    # sensor_data = df.iloc[:,:].values
    # sensor_data = np.nan_to_num(sensor_data)
    # print(df.shape)
    df.fillna(0)
    all_data.append(df)
combined_data = pd.concat(all_data, axis=0)
# print(combined_data.shape)
combined_data_sorted = combined_data.sort_values(by='label', ascending=True)
label_df =  combined_data_sorted.iloc[:,-1]
row_len = label_df.value_counts()
# print(row_len[1])
# PCA降维到2维
pca = PCA(n_components=2)
X_pca_2 = pca.fit_transform(combined_data_sorted.iloc[:,:-1])
print(combined_data_sorted)
# 绘制PCA结果的二维散点图
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111)

ax2.set_title('2D PCA of Sensor Data from Different CSV Files')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

# 为每个数据集绘制散点图
offset = 0
for i, label in enumerate(labels):
    print("i:",i)
    # 计算当前数据集的样本数量
    num_samples = row_len[i]
    # 绘制当前数据集的散点图
    ax2.scatter(X_pca_2[offset:offset+num_samples, 0], 
              X_pca_2[offset:offset+num_samples, 1], 
              c=f'C{i}', label=label)
    # 更新偏移量
    offset += num_samples

# 执行PCA降维到3维
pca = PCA(n_components=3)
X_pca = pca.fit_transform(combined_data_sorted.iloc[:,:-1])


# 绘制PCA结果的三维散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 为每个数据集绘制散点图
offset = 0
for i, label in enumerate(labels):
    # 计算当前数据集的样本数量
    num_samples = row_len[i]
    # 绘制当前数据集的散点图
    ax.scatter(X_pca[offset:offset+num_samples, 0], 
              X_pca[offset:offset+num_samples, 1], 
              X_pca[offset:offset+num_samples, 2], 
              c=f'C{i}', label=label)
    # 更新偏移量
    offset += num_samples

ax.set_title('3D PCA of Sensor Data from Different CSV Files')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()

plt.show()