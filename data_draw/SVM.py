import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA

# 设置全局字体为 Times New Roman
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 20})

# 读取数据并处理缺失值
def read_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.DataFrame(np.nan_to_num(df.values), columns=['f%d' % i for i in range(2)] + ['t', 'h'] + ['C%d' % i for i in range(32)] + ['u%d' % i for i in range(4)])
    return df

# 读取数据
orange_A = read_and_process_data('E-nose data/orange_A_14400.csv')
peach_A = read_and_process_data('E-nose data/peach_A_14400.csv')

# 提取特征和标签
X_orange = orange_A[['C%d' % i for i in range(32)]].values
X_orange = X_orange[1800:]
X_peach = peach_A[['C%d' % i for i in range(32)]].values
X_peach = X_peach[1800:]
y_orange = np.zeros(X_orange.shape[0])
y_peach = np.ones(X_peach.shape[0])

# 合并数据
X = np.vstack((X_orange, X_peach))
y = np.hstack((y_orange, y_peach))

# 使用PCA进行降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用SVM和高斯核进行分类
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X_pca, y)

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
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

plot_decision_boundary(clf, X_pca, y)
