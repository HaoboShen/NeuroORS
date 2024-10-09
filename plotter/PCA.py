import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义CSV文件路径
file_paths = ['apple1_diff.csv', 'orange1_diff.csv', 'peach1_diff.csv', 'air1_diff.csv', 'Apple2_diff.csv']
# file_paths = ['apple_2500.csv', 'orange_2500.csv', 'peach_2500.csv', 'air_2500.csv']
# 数据集标签
labels = ['Apple1', 'Orange', 'Peach', 'Air', 'Apple2']

# 读取每个CSV文件并执行PCA，同时确保所有结果具有相同的形状
min_samples = None
pca_results = []
all_data = np.array([])
row_len = []
for file_path in file_paths:
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 提取第五列开始的24个传感器数据列
    sensor_data = df.iloc[:, 4:28].values
    sensor_data = np.nan_to_num(sensor_data)
    row_len.append(sensor_data.shape[0])
    print(row_len)
    if all_data.shape[0] == 0:
        all_data = sensor_data
    else:
        all_data = np.vstack((all_data,sensor_data)) 
print(all_data, all_data.shape)
# PCA降维到2维
pca = PCA(n_components=2)
X_pca_2 = pca.fit_transform(all_data)
# 绘制PCA结果的二维散点图
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111)

ax2.set_title('2D PCA of Sensor Data from Different CSV Files')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

# 为每个数据集绘制散点图
offset = 0
for i, label in enumerate(labels):
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
X_pca = pca.fit_transform(all_data)


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