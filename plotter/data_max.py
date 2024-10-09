import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

peach_flag = [1,1]
orange_flag = [1,1]
apple_flag = [1,1]
air_flag = [1,1]

peach1 = pd.read_csv('peach_2500.csv')
peach2 = pd.read_csv('peach_2500_20240410.csv')
apple1 = pd.read_csv('apple_2500.csv')
apple2 = pd.read_csv('apple_2500_20240410.csv')
orange1 = pd.read_csv('orange_2500.csv')
orange2 = pd.read_csv('orange_2500_20240410.csv')
air1 = pd.read_csv('air_2500.csv')
air2 = pd.read_csv('air_2500_20240410.csv')

combined_df = pd.DataFrame()
print(peach1.shape, peach2.shape, apple1.shape, apple2.shape, orange1.shape, orange2.shape, air1.shape, air2.shape)

# 第一行加入列名
peach1 = pd.DataFrame(peach1.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
peach2 = pd.DataFrame(peach2.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
apple1 = pd.DataFrame(apple1.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
apple2 = pd.DataFrame(apple2.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
orange1 = pd.DataFrame(orange1.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
orange2 = pd.DataFrame(orange2.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
air1 = pd.DataFrame(air1.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
air2 = pd.DataFrame(air2.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])


peach1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
peach2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
apple1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
apple2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
orange1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
orange2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
air1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
air2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)

peach1_max = peach1.max()
peach2_max = peach2.max()
apple1_max = apple1.max()
apple2_max = apple2.max()
orange1_max = orange1.max()
orange2_max = orange2.max()
air1_max = air1.max()
air2_max = air2.max()

# 创建一个新的DataFrame，用于存储两个数据集的均值
maxs_combined = pd.DataFrame({
    'peach1': peach1_max,
    'peach2': peach2_max,
    'apple1': apple1_max,
    'apple2': apple2_max,
    'orange1': orange1_max,
    'orange2': orange2_max,
    'air1': air1_max,
    'air2': air2_max
})

# 为每个列绘制图形
if peach_flag[0] == 1:
    plt.plot(maxs_combined.index, maxs_combined['peach1'], label='peach 1' , color='pink')
if peach_flag[1] == 1:
    plt.plot(maxs_combined.index, maxs_combined['peach2'], label='peach 2', color='purple')
if apple_flag[0] == 1:
    plt.plot(maxs_combined.index, maxs_combined['apple1'], label='apple 1' , color='red')
if apple_flag[1] == 1:
    plt.plot(maxs_combined.index, maxs_combined['apple2'], label='apple 2', color='green')
if orange_flag[0] == 1:
    plt.plot(maxs_combined.index, maxs_combined['orange1'], label='orange 1', color='orange')
if orange_flag[1] == 1:
    plt.plot(maxs_combined.index, maxs_combined['orange2'], label='orange 2', color='yellow')
if air_flag[0] == 1:
    plt.plot(maxs_combined.index, maxs_combined['air1'], label='air 1', color='blue')
if air_flag[1] == 1:
    plt.plot(maxs_combined.index, maxs_combined['air2'], label='air 2', color='black')

# 设置图形标题和图例
plt.title('max Values Comparison for Each Column')
plt.legend()

# 显示图形
plt.show()