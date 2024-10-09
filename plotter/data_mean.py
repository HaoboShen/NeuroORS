import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

peach_flag = [1,0]
orange_flag = [1,0]
apple_flag = [0,0]
air_flag = [0,0]
banana_flag = [0,0]
watermelon_flag = [0,0]
alcohol1_flag = [0,0]
vinegar1_flag = [0,0]

peach1 = pd.read_csv('peach_mix21_B_2501.csv')
peach2 = pd.read_csv('peach_2500.csv')
apple1 = pd.read_csv('apple_2500.csv')
apple2 = pd.read_csv('apple_2500.csv')
orange1 = pd.read_csv('orange_mix21_B_2580.csv')
orange2 = pd.read_csv('orange_2500.csv')
air1 = pd.read_csv('air_2500.csv')
air2 = pd.read_csv('air_2500.csv')
banana1 = pd.read_csv('banana_5376.csv')
watermelon1 = pd.read_csv('watermelon_3458.csv')
watermelon2 = pd.read_csv('A_watermelon_2925.csv')
alcohol1 = pd.read_csv('A_alcohol_2979.csv')
vinegar1 = pd.read_csv('A_vinegar_2604.csv')
# peach1 = pd.read_csv('peach1_diff.csv')
# peach2 = pd.read_csv('peach_diff_20240417.csv')
# apple1 = pd.read_csv('apple1_diff.csv')
# apple2 = pd.read_csv('apple_diff_20240417.csv')
# orange1 = pd.read_csv('orange1_diff.csv')
# orange2 = pd.read_csv('orange_diff_20240417.csv')
# air1 = pd.read_csv('air1_diff.csv')
# air2 = pd.read_csv('air_diff_20240417.csv')

combined_df = pd.DataFrame()
print(peach1.shape, peach2.shape, apple1.shape, apple2.shape, orange1.shape, orange2.shape, air1.shape, air2.shape)

# 第一行加入列名
peach1 = pd.DataFrame(np.nan_to_num(peach1.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
peach2 = pd.DataFrame(np.nan_to_num(peach2.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
apple1 = pd.DataFrame(np.nan_to_num(apple1.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
apple2 = pd.DataFrame(np.nan_to_num(apple2.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
orange1 = pd.DataFrame(np.nan_to_num(orange1.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
orange2 = pd.DataFrame(np.nan_to_num(orange2.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
air1 = pd.DataFrame(np.nan_to_num(air1.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
air2 = pd.DataFrame(np.nan_to_num(air2.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
banana1 = pd.DataFrame(np.nan_to_num(banana1.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
watermelon1 = pd.DataFrame(np.nan_to_num(watermelon1.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
watermelon2 = pd.DataFrame(np.nan_to_num(watermelon2.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
alcohol1 = pd.DataFrame(np.nan_to_num(alcohol1.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
vinegar1 = pd.DataFrame(np.nan_to_num(vinegar1.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])

peach1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
peach2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
apple1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
apple2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
orange1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
orange2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
air1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
air2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
banana1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
watermelon1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
watermelon2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
alcohol1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
vinegar1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)

peach1_mean = peach1.mean()
peach2_mean = peach2.mean()
apple1_mean = apple1.mean()
apple2_mean = apple2.mean()
orange1_mean = orange1.mean()
orange2_mean = orange2.mean()
air1_mean = air1.mean()
air2_mean = air2.mean()
banana1_mean = banana1.mean()
watermelon1_mean = watermelon1.mean()
watermelon2_mean = watermelon2.mean()
alcohol1_mean = alcohol1.mean()
vinegar1_mean = vinegar1.mean()

# 创建一个新的DataFrame，用于存储两个数据集的均值
means_combined = pd.DataFrame({
    'peach1': peach1_mean,
    'peach2': peach2_mean,
    'apple1': apple1_mean,
    'apple2': apple2_mean,
    'orange1': orange1_mean,
    'orange2': orange2_mean,
    'air1': air1_mean,
    'air2': air2_mean,
    'banana1': banana1_mean,
    'watermelon1': watermelon1_mean,
    'watermelon2': watermelon2_mean,
    'alcohol1': alcohol1_mean,
    'vinegar1': vinegar1_mean
})

# 为每个列绘制图形
if peach_flag[0] == 1:
    plt.plot(means_combined.index, means_combined['peach1'], label='peach 1' , color='pink')
if peach_flag[1] == 1:
    plt.plot(means_combined.index, means_combined['peach2'], label='peach 2', color='purple')
if apple_flag[0] == 1:
    plt.plot(means_combined.index, means_combined['apple1'], label='apple 1' , color='red')
if apple_flag[1] == 1:
    plt.plot(means_combined.index, means_combined['apple2'], label='apple 2', color='green')
if orange_flag[0] == 1:
    plt.plot(means_combined.index, means_combined['orange1'], label='orange 1', color='orange')
if orange_flag[1] == 1:
    plt.plot(means_combined.index, means_combined['orange2'], label='orange 2', color='red')
if air_flag[0] == 1:
    plt.plot(means_combined.index, means_combined['air1'], label='air 1', color='blue')
if air_flag[1] == 1:
    plt.plot(means_combined.index, means_combined['air2'], label='air 2', color='black')
if banana_flag[0] == 1:
    plt.plot(means_combined.index, means_combined['banana1'], label='banana 1', color='yellow')

if watermelon_flag[0] == 1:
    plt.plot(means_combined.index, means_combined['watermelon1'], label='watermelon 1', color='red')
if watermelon_flag[1] == 1:
    plt.plot(means_combined.index, means_combined['watermelon2'], label='watermelon 2', color='green')
if alcohol1_flag[0] == 1:
    plt.plot(means_combined.index, means_combined['alcohol1'], label='alcohol 1', color='blue')
if vinegar1_flag[0] == 1:
    plt.plot(means_combined.index, means_combined['vinegar1'], label='vinegar 1', color='black')

# 设置图形标题和图例
plt.title('Mean Values Comparison for Each Column')
plt.legend()

# 显示图形
plt.show()