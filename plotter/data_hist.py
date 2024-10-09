import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('apple1_diff.csv')
df2 = pd.read_csv('peach1_diff.csv')
combined_df = pd.DataFrame()
print(df1.shape, df2.shape)

# # 第一行加入列名
# df1 = pd.DataFrame(df1.values,columns = ['flag%d'%i for i in range(2)]+['temperature','humidity']+['CH%d'%i for i in range(32)]+['unused%d'%i for i in range(4)])
# df2 = pd.DataFrame(df2.values,columns = ['flag%d'%i for i in range(2)]+['temperature','humidity']+['CH%d'%i for i in range(32)]+['unused%d'%i for i in range(4)])

# 遍历两个DataFrame中的所有列
for column in df1.columns:
    # 创建一个新图形和轴对象
    fig, ax = plt.subplots()
    
    # 绘制第一个数据集的直方图
    sns.histplot(df1[column], bins=30, kde=False, alpha=0.5, ax=ax, label='Data 1')
    
    # 绘制第二个数据集的直方图
    sns.histplot(df2[column], bins=30, kde=False, alpha=0.5, ax=ax, label='Data 2')
    
    # 设置图形标题
    ax.set_title(f'Histogram Comparison for {column}')

    plt.legend(loc='upper right')

# 显示所有图形
plt.show()