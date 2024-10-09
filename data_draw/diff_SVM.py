import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# 设置全局字体为 Times New Roman
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 20})

# 读取数据并处理缺失值
def read_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.DataFrame(np.nan_to_num(df.values), columns=['f%d' % i for i in range(2)] + ['t', 'h'] + ['C%d' % i for i in range(32)] + ['u%d' % i for i in range(4)])
    return df

# 差分处理函数
def apply_diff(df):
    diff_df = df.copy()
    for i in range(32):
        diff_df['C%d' % i] = np.diff(df['C%d' % i], prepend=df['C%d' % i][0])
    return diff_df

# 读取数据
orange_A = read_and_process_data('E-nose data/orange_A_14400.csv')
peach_A = read_and_process_data('E-nose data/peach_A_14400.csv')

# 对数据进行差分处理
orange_A_diff = apply_diff(orange_A)
peach_A_diff = apply_diff(peach_A)

# 提取特征和标签
X_orange = orange_A_diff[['C%d' % i for i in range(32)]].values
X_peach = peach_A_diff[['C%d' % i for i in range(32)]].values
y_orange = np.zeros(X_orange.shape[0])
y_peach = np.ones(X_peach.shape[0])

# 合并数据
X = np.vstack((X_orange, X_peach))
y = np.hstack((y_orange, y_peach))

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用SVM和高斯核进行分类
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X_scaled, y)

# 绘制决策边界
def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title('SVM with RBF Kernel Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 由于数据维度较高（32维），为了绘制决策边界，我们可以选择前两维进行可视化
X_vis = X_scaled[:, :2]
clf_vis = svm.SVC(kernel='rbf', gamma='scale')
clf_vis.fit(X_vis, y)
plot_decision_boundary(clf_vis, X_vis, y)
