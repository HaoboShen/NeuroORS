import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
 
def lowpass_filter(data, fs, cutoff_freq, order=5):
    # 计算截止频率对应的截止索引
    cutoff_index = int(cutoff_freq * len(data) / fs)
    
    # 设计低通滤波器
    b, a = butter(order, cutoff_index / (len(data) / 2.0), btype='low', analog=False)
    # print(b, a)
    # 应用滤波器
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

peach_flag = [0,0]
orange_flag = [0,0]
apple_flag = [0,0]
air_flag = [0,0]
watermelon_flag = [0,0]
alcohol_flag = [1,1]
vinegar_flag = [1,1]
save_flag = 1

filter_flag = 1

# peach_flag = [1,1]
# orange_flag = [1,1]
# apple_flag = [1,1]
# air_flag = [1,1]
# save_flag = 1

colors_peach1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_peach2 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_apple1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_apple2 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_orange1 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']
colors_orange2 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']
colors_air1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_air2 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']
colors_watermelon1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_watermelon2 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']
colors_alchohol1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_alchohol2 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']
colors_vinegar1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_vinegar2 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']

peach1 = pd.read_csv('peach_2500.csv')
peach2 = pd.read_csv('peach_2500.csv')
apple1 = pd.read_csv('apple_2500.csv')
apple2 = pd.read_csv('apple_2500.csv')
orange1 = pd.read_csv('orange_2500.csv')
orange2 = pd.read_csv('orange_2500.csv')
air1 = pd.read_csv('air_2500.csv')
air2 = pd.read_csv('air_2500.csv')
watermelon1 = pd.read_csv('watermelon_3458.csv')
watermelon2 = pd.read_csv('A_watermelon_2925.csv')
alcohol1 = pd.read_csv('A_alcohol_2979.csv')
alcohol2 = pd.read_csv('A_alcohol_2979.csv')
vinegar1 = pd.read_csv('A_vinegar_2604.csv')
vinegar2 = pd.read_csv('A_vinegar_2604.csv')

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
watermelon1 = pd.DataFrame(watermelon1.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
watermelon2 = pd.DataFrame(watermelon2.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
alcohol1 = pd.DataFrame(alcohol1.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
alcohol2 = pd.DataFrame(alcohol2.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
vinegar1 = pd.DataFrame(vinegar1.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
vinegar2 = pd.DataFrame(vinegar2.values,columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])

if save_flag == 0:
    peach1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    peach2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    apple1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    apple2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    orange1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    orange2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    air1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    air2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    watermelon1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    watermelon2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    alcohol1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    alcohol2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    vinegar1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    vinegar2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)


# for feature_index in range(32):

#     # 初始化卡尔曼滤波器参数
#     n_dim_state = 1  # 状态维度为1
#     n_dim_obs = 1  # 观测维度为1
#     initial_state_mean = 0  # 初始状态估计为0
#     transition_matrix = np.array([[1]])  # 状态转移矩阵
#     observation_matrix = np.array([[1]])  # 观测矩阵
#     process_noise_cov = np.array([[0.01]])  # 过程噪声协方差
#     observation_noise_cov = np.array([[0.01]])  # 观测噪声协方差

#     # 创建卡尔曼滤波器实例
#     kf = pk.KalmanFilter(
#         initial_state_mean=initial_state_mean,
#         initial_state_covariance=process_noise_cov,
#         observation_covariance=observation_noise_cov,
#         transition_matrices=transition_matrix,
#         observation_matrices=observation_matrix
#     )

#     # 对选定的特征进行滤波
#     filtered_data = []
#     for idx, value in enumerate(apple1.iloc[:, feature_index]):
#         # 由于第一个值没有前一个值，我们不能对其进行滤波
#         if idx == 0:
#             filtered_data.append(value)
#         else:
#             # 应用卡尔曼滤波器
#             state, _ = kf.filter(value)
#             filtered_data.append(state[0])

#     # 将滤波后的数据添加到DataFrame中
#     apple1['C%d'%feature_index] = filtered_data

    # # 绘制原始和滤波后的数据
    # plt.figure()
    # plt.plot(apple1.index, apple1.iloc[:, feature_index], label='Original Data')
    # plt.plot(apple1.index, apple1['C%d'%feature_index], label='Filtered Data', linestyle='--')
    # plt.legend()
    # plt.show()
# 低通滤波
if filter_flag == 1:
    fs = 1
    cutoff_freq = 0.005
    for i in range(32):
        peach1['C%d'%i] = lowpass_filter(np.nan_to_num(peach1['C%d'%i]), fs, cutoff_freq)
        peach2['C%d'%i] = lowpass_filter(np.nan_to_num(peach2['C%d'%i]), fs, cutoff_freq)
        apple1['C%d'%i] = lowpass_filter(np.nan_to_num(apple1['C%d'%i]), fs, cutoff_freq)
        apple2['C%d'%i] = lowpass_filter(np.nan_to_num(apple2['C%d'%i]), fs, cutoff_freq)
        orange1['C%d'%i] = lowpass_filter(np.nan_to_num(orange1['C%d'%i]), fs, cutoff_freq)
        orange2['C%d'%i] = lowpass_filter(np.nan_to_num(orange2['C%d'%i]), fs, cutoff_freq)
        air1['C%d'%i] = lowpass_filter(np.nan_to_num(air1['C%d'%i]), fs, cutoff_freq)
        air2['C%d'%i] = lowpass_filter(np.nan_to_num(air2['C%d'%i]), fs, cutoff_freq)
        watermelon1['C%d'%i] = lowpass_filter(np.nan_to_num(watermelon1['C%d'%i]), fs, cutoff_freq)
        watermelon2['C%d'%i] = lowpass_filter(np.nan_to_num(watermelon2['C%d'%i]), fs, cutoff_freq)
        alcohol1['C%d'%i] = lowpass_filter(np.nan_to_num(alcohol1['C%d'%i]), fs, cutoff_freq)
        alcohol2['C%d'%i] = lowpass_filter(np.nan_to_num(alcohol2['C%d'%i]), fs, cutoff_freq)
        vinegar1['C%d'%i] = lowpass_filter(np.nan_to_num(vinegar1['C%d'%i]), fs, cutoff_freq)
        vinegar2['C%d'%i] = lowpass_filter(np.nan_to_num(vinegar2['C%d'%i]), fs, cutoff_freq)


# 对每个数据做差分
peach1_diff = peach1.diff()
peach2_diff = peach2.diff()
apple1_diff = apple1.diff()
apple2_diff = apple2.diff()
orange1_diff = orange1.diff()
orange2_diff = orange2.diff()
air1_diff = air1.diff()
air2_diff = air2.diff()
watermelon1_diff = watermelon1.diff()
watermelon2_diff = watermelon2.diff()
alcohol1_diff = alcohol1.diff()
alcohol2_diff = alcohol2.diff()
vinegar1_diff = vinegar1.diff()
vinegar2_diff = vinegar2.diff()

# # 低通滤波
# if filter_flag == 1:
#     fs = 1
#     cutoff_freq = 0.005
#     for i in range(32):
#         peach1_diff['C%d'%i] = lowpass_filter(np.nan_to_num(peach1_diff['C%d'%i]), fs, cutoff_freq)
#         peach2_diff['C%d'%i] = lowpass_filter(np.nan_to_num(peach2_diff['C%d'%i]), fs, cutoff_freq)
#         apple1_diff['C%d'%i] = lowpass_filter(np.nan_to_num(apple1_diff['C%d'%i]), fs, cutoff_freq)
#         print(apple1_diff['C%d'%i])
#         apple2_diff['C%d'%i] = lowpass_filter(np.nan_to_num(apple2_diff['C%d'%i]), fs, cutoff_freq)
#         orange1_diff['C%d'%i] = lowpass_filter(np.nan_to_num(orange1_diff['C%d'%i]), fs, cutoff_freq)
#         orange2_diff['C%d'%i] = lowpass_filter(np.nan_to_num(orange2_diff['C%d'%i]), fs, cutoff_freq)
#         air1_diff['C%d'%i] = lowpass_filter(np.nan_to_num(air1_diff['C%d'%i]), fs, cutoff_freq)
#         air2_diff['C%d'%i] = lowpass_filter(np.nan_to_num(air2_diff['C%d'%i]), fs, cutoff_freq)


# # 取绝对值
# peach1_diff = peach1_diff.abs()
# peach2_diff = peach2_diff.abs()
# apple1_diff = apple1_diff.abs()
# apple2_diff = apple2_diff.abs()
# orange1_diff = orange1_diff.abs()
# orange2_diff = orange2_diff.abs()
# air1_diff = air1_diff.abs()
# air2_diff = air2_diff.abs()

# # 去除绝对值小于0.005的值
# th = 0.0005
# peach1_diff = peach1_diff[peach1_diff.abs() > th]
# peach2_diff = peach2_diff[peach2_diff.abs() > th]
# apple1_diff = apple1_diff[apple1_diff.abs() > th]
# apple2_diff = apple2_diff[apple2_diff.abs() > th]
# orange1_diff = orange1_diff[orange1_diff.abs() > th]
# orange2_diff = orange2_diff[orange2_diff.abs() > th]
# air1_diff = air1_diff[air1_diff.abs() > th]
# air2_diff = air2_diff[air2_diff.abs() > th]

# 保存到csv
if save_flag == 1:
    peach1_diff.to_csv('peach1_diff_20240424.csv')
    peach2_diff.to_csv('peach_diff_20240424.csv')
    apple1_diff.to_csv('apple1_diff_20240424.csv')
    apple2_diff.to_csv('apple_diff_20240424.csv')
    orange1_diff.to_csv('orange1_diff_20240424.csv')
    orange2_diff.to_csv('orange_diff_20240424.csv')
    air1_diff.to_csv('air1_diff_20240424.csv')
    air2_diff.to_csv('air_diff_20240424.csv')
    watermelon1_diff.to_csv('watermelon1_diff_20240424.csv')
    watermelon2_diff.to_csv('watermelon_diff_20240424.csv')
    alcohol1_diff.to_csv('alcohol1_diff_20240424.csv')
    alcohol2_diff.to_csv('alcohol2_diff_20240424.csv')
    vinegar1_diff.to_csv('vinegar1_diff_20240424.csv')
    vinegar2_diff.to_csv('vinegar2_diff_20240424.csv')

# 为每个列绘制图形
if peach_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(peach1_diff.index, peach1_diff['C%d'%i], label='peach1_C%d'%(i+1) , color=colors_peach1[i])
        # 设置图形标题
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(peach1_diff.index, peach1_diff['C%d'%(i+8)], label='peach1_C%d'%(i+9) , color=colors_peach1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(peach1_diff.index, peach1_diff['C%d'%(i+16)], label='peach1_C%d'%(i+17) , color=colors_peach1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(peach1_diff.index, peach1_diff['C%d'%(i+24)], label='peach1_C%d'%(i+25) , color=colors_peach1[i])
        plt.title('C25-C32')
        plt.legend()
if peach_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(peach2_diff.index, peach2_diff['C%d'%i], label='peach2_C%d'%(i+1) , color=colors_peach2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(peach2_diff.index, peach2_diff['C%d'%(i+8)], label='peach2_C%d'%(i+9) , color=colors_peach2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(peach2_diff.index, peach2_diff['C%d'%(i+16)], label='peach2_C%d'%(i+17) , color=colors_peach2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(peach2_diff.index, peach2_diff['C%d'%(i+24)], label='peach2_C%d'%(i+25) , color=colors_peach2[i])
        plt.title('C25-C32')
        plt.legend()
if apple_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(apple1_diff.index, apple1_diff['C%d'%i], label='apple1_C%d'%(i+1) , color=colors_apple1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(apple1_diff.index, apple1_diff['C%d'%(i+8)], label='apple1_C%d'%(i+9) , color=colors_apple1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(apple1_diff.index, apple1_diff['C%d'%(i+16)], label='apple1_C%d'%(i+17) , color=colors_apple1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(apple1_diff.index, apple1_diff['C%d'%(i+24)], label='apple1_C%d'%(i+25) , color=colors_apple1[i])
        plt.title('C25-C32')
        plt.legend()
if apple_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(apple2_diff.index, apple2_diff['C%d'%i], label='apple2_C%d'%(i+1) , color=colors_apple2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(apple2_diff.index, apple2_diff['C%d'%(i+8)], label='apple2_C%d'%(i+9) , color=colors_apple2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(apple2_diff.index, apple2_diff['C%d'%(i+16)], label='apple2_C%d'%(i+17) , color=colors_apple2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(apple2_diff.index, apple2_diff['C%d'%(i+24)], label='apple2_C%d'%(i+25) , color=colors_apple2[i])
        plt.title('C25-C32')
        plt.legend()
    
if orange_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(orange1_diff.index, orange1_diff['C%d'%i], label='orange1_C%d'%(i+1) , color=colors_orange1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(orange1_diff.index, orange1_diff['C%d'%(i+8)], label='orange1_C%d'%(i+9) , color=colors_orange1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(orange1_diff.index, orange1_diff['C%d'%(i+16)], label='orange1_C%d'%(i+17) , color=colors_orange1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(orange1_diff.index, orange1_diff['C%d'%(i+24)], label='orange1_C%d'%(i+25) , color=colors_orange1[i])
        plt.title('C25-C32')
        plt.legend()
if orange_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(orange2_diff.index, orange2_diff['C%d'%i], label='orange2_C%d'%(i+1) , color=colors_orange2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(orange2_diff.index, orange2_diff['C%d'%(i+8)], label='orange2_C%d'%(i+9) , color=colors_orange2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(orange2_diff.index, orange2_diff['C%d'%(i+16)], label='orange2_C%d'%(i+17) , color=colors_orange2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(orange2_diff.index, orange2_diff['C%d'%(i+24)], label='orange2_C%d'%(i+25) , color=colors_orange2[i])
        plt.title('C25-C32')
        plt.legend()
if air_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(air1_diff.index, air1_diff['C%d'%i], label='air1_C%d'%(i+1) , color=colors_air1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(air1_diff.index, air1_diff['C%d'%(i+8)], label='air1_C%d'%(i+9) , color=colors_air1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(air1_diff.index, air1_diff['C%d'%(i+16)], label='air1_C%d'%(i+17) , color=colors_air1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(air1_diff.index, air1_diff['C%d'%(i+24)], label='air1_C%d'%(i+25) , color=colors_air1[i])
        plt.title('C25-C32')
        plt.legend()
if air_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(air2_diff.index, air2_diff['C%d'%i], label='air2_C%d'%(i+1) , color=colors_air2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(air2_diff.index, air2_diff['C%d'%(i+8)], label='air2_C%d'%(i+9) , color=colors_air2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(air2_diff.index, air2_diff['C%d'%(i+16)], label='air2_C%d'%(i+17) , color=colors_air2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(air2_diff.index, air2_diff['C%d'%(i+24)], label='air2_C%d'%(i+25) , color=colors_air2[i])
        plt.title('C25-C32')
        plt.legend()
if watermelon_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(watermelon1_diff.index, watermelon1_diff['C%d'%i], label='watermelon1_C%d'%(i+1) , color=colors_watermelon1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(watermelon1_diff.index, watermelon1_diff['C%d'%(i+8)], label='watermelon1_C%d'%(i+9) , color=colors_watermelon1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(watermelon1_diff.index, watermelon1_diff['C%d'%(i+16)], label='watermelon1_C%d'%(i+17) , color=colors_watermelon1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(watermelon1_diff.index, watermelon1_diff['C%d'%(i+24)], label='watermelon1_C%d'%(i+25) , color=colors_watermelon1[i])
        plt.title('C25-C32')
        plt.legend()
if watermelon_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(watermelon2_diff.index, watermelon2_diff['C%d'%i], label='watermelon2_C%d'%(i+1) , color=colors_watermelon2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(watermelon2_diff.index, watermelon2_diff['C%d'%(i+8)], label='watermelon2_C%d'%(i+9) , color=colors_watermelon2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(watermelon2_diff.index, watermelon2_diff['C%d'%(i+16)], label='watermelon2_C%d'%(i+17) , color=colors_watermelon2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(watermelon2_diff.index, watermelon2_diff['C%d'%(i+24)], label='watermelon2_C%d'%(i+25) , color=colors_watermelon2[i])
        plt.title('C25-C32')
        plt.legend()
if alcohol_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(alcohol1_diff.index, alcohol1_diff['C%d'%i], label='alcohol1_C%d'%(i+1) , color=colors_alchohol1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(alcohol1_diff.index, alcohol1_diff['C%d'%(i+8)], label='alcohol1_C%d'%(i+9) , color=colors_alchohol1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(alcohol1_diff.index, alcohol1_diff['C%d'%(i+16)], label='alcohol1_C%d'%(i+17) , color=colors_alchohol1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(alcohol1_diff.index, alcohol1_diff['C%d'%(i+24)], label='alcohol1_C%d'%(i+25) , color=colors_alchohol1[i])
        plt.title('C25-C32')
        plt.legend()
if alcohol_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(alcohol2_diff.index, alcohol2_diff['C%d'%i], label='alcohol2_C%d'%(i+1) , color=colors_alchohol2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(alcohol2_diff.index, alcohol2_diff['C%d'%(i+8)], label='alcohol2_C%d'%(i+9) , color=colors_alchohol2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(alcohol2_diff.index, alcohol2_diff['C%d'%(i+16)], label='alcohol2_C%d'%(i+17) , color=colors_alchohol2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(alcohol2_diff.index, alcohol2_diff['C%d'%(i+24)], label='alcohol2_C%d'%(i+25) , color=colors_alchohol2[i])
        plt.title('C25-C32')
        plt.legend()
if vinegar_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(vinegar1_diff.index, vinegar1_diff['C%d'%i], label='vinegar1_C%d'%(i+1) , color=colors_vinegar1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(vinegar1_diff.index, vinegar1_diff['C%d'%(i+8)], label='vinegar1_C%d'%(i+9) , color=colors_vinegar1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(vinegar1_diff.index, vinegar1_diff['C%d'%(i+16)], label='vinegar1_C%d'%(i+17) , color=colors_vinegar1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(vinegar1_diff.index, vinegar1_diff['C%d'%(i+24)], label='vinegar1_C%d'%(i+25) , color=colors_vinegar1[i])
        plt.title('C25-C32')
        plt.legend()
if vinegar_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(vinegar2_diff.index, vinegar2_diff['C%d'%i], label='vinegar2_C%d'%(i+1) , color=colors_vinegar2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(vinegar2_diff.index, vinegar2_diff['C%d'%(i+8)], label='vinegar2_C%d'%(i+9) , color=colors_vinegar2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(vinegar2_diff.index, vinegar2_diff['C%d'%(i+16)], label='vinegar2_C%d'%(i+17) , color=colors_vinegar2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(vinegar2_diff.index, vinegar2_diff['C%d'%(i+24)], label='vinegar2_C%d'%(i+25) , color=colors_vinegar2[i])
        plt.title('C25-C32')
        plt.legend()

# 显示图形
plt.show()