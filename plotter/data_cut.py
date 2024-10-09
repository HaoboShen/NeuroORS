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

# peach_flag = [0,1]
# orange_flag = [0,0]
# apple_flag = [0,0]
# air_flag = [0,0]
# save_flag = 0

filter_flag = 1

peach_flag = [1,1]
orange_flag = [1,1]
apple_flag = [1,1]
air_flag = [1,1]
save_flag = 1

colors_peach1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_peach2 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_apple1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_apple2 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_orange1 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']
colors_orange2 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']
colors_air1 = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'black', 'brown']
colors_air2 = ['pink','lightgreen','lightblue','#EECFFF','#FFDAB9','#FFFFE0','gray','#D2B48C']

peach1 = pd.read_csv('peach_2500.csv')
peach2 = pd.read_csv('peach_2500_20240410.csv')
apple1 = pd.read_csv('apple_2500.csv')
apple2 = pd.read_csv('apple_2500_20240410.csv')
orange1 = pd.read_csv('orange_2500.csv')
orange2 = pd.read_csv('orange_2500_20240410.csv')
air1 = pd.read_csv('air_2500.csv')
air2 = pd.read_csv('air_2500_20240410.csv')

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

if save_flag == 0:
    peach1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    peach2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    apple1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    apple2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    orange1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    orange2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    air1.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)
    air2.drop(['f%d'%i for i in range(2)]+['t','h']+['u%d'%i for i in range(4)], axis=1, inplace=True)


# # 对每个数据做差
# peach1 = peach1.diff()
# peach2 = peach2.diff()
# apple1 = apple1.diff()
# apple2 = apple2.diff()
# orange1 = orange1.diff()
# orange2 = orange2.diff()
# air1 = air1.diff()
# air2 = air2.diff()
#取出前1000个数据
peach1 = peach1[:1000]
peach2 = peach2[:1000]
apple1 = apple1[:1000]
apple2 = apple2[:1000]
orange1 = orange1[:1000]
orange2 = orange2[:1000]
air1 = air1[:1000]
air2 = air2[:1000]

# # 低通滤波
# if filter_flag == 1:
#     fs = 1
#     cutoff_freq = 0.005
#     for i in range(32):
#         peach1['C%d'%i] = lowpass_filter(np.nan_to_num(peach1['C%d'%i]), fs, cutoff_freq)
#         peach2['C%d'%i] = lowpass_filter(np.nan_to_num(peach2['C%d'%i]), fs, cutoff_freq)
#         apple1['C%d'%i] = lowpass_filter(np.nan_to_num(apple1['C%d'%i]), fs, cutoff_freq)
#         print(apple1['C%d'%i])
#         apple2['C%d'%i] = lowpass_filter(np.nan_to_num(apple2['C%d'%i]), fs, cutoff_freq)
#         orange1['C%d'%i] = lowpass_filter(np.nan_to_num(orange1['C%d'%i]), fs, cutoff_freq)
#         orange2['C%d'%i] = lowpass_filter(np.nan_to_num(orange2['C%d'%i]), fs, cutoff_freq)
#         air1['C%d'%i] = lowpass_filter(np.nan_to_num(air1['C%d'%i]), fs, cutoff_freq)
#         air2['C%d'%i] = lowpass_filter(np.nan_to_num(air2['C%d'%i]), fs, cutoff_freq)


# # 取绝对值
# peach1 = peach1.abs()
# peach2 = peach2.abs()
# apple1 = apple1.abs()
# apple2 = apple2.abs()
# orange1 = orange1.abs()
# orange2 = orange2.abs()
# air1 = air1.abs()
# air2 = air2.abs()

# # 去除绝对值小于0.005的值
# th = 0.0005
# peach1 = peach1[peach1.abs() > th]
# peach2 = peach2[peach2.abs() > th]
# apple1 = apple1[apple1.abs() > th]
# apple2 = apple2[apple2.abs() > th]
# orange1 = orange1[orange1.abs() > th]
# orange2 = orange2[orange2.abs() > th]
# air1 = air1[air1.abs() > th]
# air2 = air2[air2.abs() > th]

# 保存到csv
if save_flag == 1:
    peach1.to_csv('peach1_cut.csv')
    peach2.to_csv('peach2_cut.csv')
    apple1.to_csv('apple1_cut.csv')
    apple2.to_csv('apple2_cut.csv')
    orange1.to_csv('orange1_cut.csv')
    orange2.to_csv('orange2_cut.csv')
    air1.to_csv('air1_cut.csv')
    air2.to_csv('air2_cut.csv')
print(air1.shape, air2.shape)


# 为每个列绘制图形
if peach_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(peach1.index, peach1['C%d'%i], label='peach1_C%d'%(i+1) , color=colors_peach1[i])
        # 设置图形标题
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(peach1.index, peach1['C%d'%(i+8)], label='peach1_C%d'%(i+9) , color=colors_peach1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(peach1.index, peach1['C%d'%(i+16)], label='peach1_C%d'%(i+17) , color=colors_peach1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(peach1.index, peach1['C%d'%(i+24)], label='peach1_C%d'%(i+25) , color=colors_peach1[i])
        plt.title('C25-C32')
        plt.legend()
if peach_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(peach2.index, peach2['C%d'%i], label='peach2_C%d'%(i+1) , color=colors_peach2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(peach2.index, peach2['C%d'%(i+8)], label='peach2_C%d'%(i+9) , color=colors_peach2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(peach2.index, peach2['C%d'%(i+16)], label='peach2_C%d'%(i+17) , color=colors_peach2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(peach2.index, peach2['C%d'%(i+24)], label='peach2_C%d'%(i+25) , color=colors_peach2[i])
        plt.title('C25-C32')
        plt.legend()
if apple_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(apple1.index, apple1['C%d'%i], label='apple1_C%d'%(i+1) , color=colors_apple1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(apple1.index, apple1['C%d'%(i+8)], label='apple1_C%d'%(i+9) , color=colors_apple1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(apple1.index, apple1['C%d'%(i+16)], label='apple1_C%d'%(i+17) , color=colors_apple1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(apple1.index, apple1['C%d'%(i+24)], label='apple1_C%d'%(i+25) , color=colors_apple1[i])
        plt.title('C25-C32')
        plt.legend()
if apple_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(apple2.index, apple2['C%d'%i], label='apple2_C%d'%(i+1) , color=colors_apple2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(apple2.index, apple2['C%d'%(i+8)], label='apple2_C%d'%(i+9) , color=colors_apple2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(apple2.index, apple2['C%d'%(i+16)], label='apple2_C%d'%(i+17) , color=colors_apple2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(apple2.index, apple2['C%d'%(i+24)], label='apple2_C%d'%(i+25) , color=colors_apple2[i])
        plt.title('C25-C32')
        plt.legend()
    
if orange_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(orange1.index, orange1['C%d'%i], label='orange1_C%d'%(i+1) , color=colors_orange1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(orange1.index, orange1['C%d'%(i+8)], label='orange1_C%d'%(i+9) , color=colors_orange1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(orange1.index, orange1['C%d'%(i+16)], label='orange1_C%d'%(i+17) , color=colors_orange1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(orange1.index, orange1['C%d'%(i+24)], label='orange1_C%d'%(i+25) , color=colors_orange1[i])
        plt.title('C25-C32')
        plt.legend()
if orange_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(orange2.index, orange2['C%d'%i], label='orange2_C%d'%(i+1) , color=colors_orange2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(orange2.index, orange2['C%d'%(i+8)], label='orange2_C%d'%(i+9) , color=colors_orange2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(orange2.index, orange2['C%d'%(i+16)], label='orange2_C%d'%(i+17) , color=colors_orange2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(orange2.index, orange2['C%d'%(i+24)], label='orange2_C%d'%(i+25) , color=colors_orange2[i])
        plt.title('C25-C32')
        plt.legend()
if air_flag[0] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(air1.index, air1['C%d'%i], label='air1_C%d'%(i+1) , color=colors_air1[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(air1.index, air1['C%d'%(i+8)], label='air1_C%d'%(i+9) , color=colors_air1[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(air1.index, air1['C%d'%(i+16)], label='air1_C%d'%(i+17) , color=colors_air1[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(air1.index, air1['C%d'%(i+24)], label='air1_C%d'%(i+25) , color=colors_air1[i])
        plt.title('C25-C32')
        plt.legend()
if air_flag[1] == 1:
    for i in range(8):
        fig1 = plt.figure(num="C1-C8")
        plt.plot(air2.index, air2['C%d'%i], label='air2_C%d'%(i+1) , color=colors_air2[i])
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(air2.index, air2['C%d'%(i+8)], label='air2_C%d'%(i+9) , color=colors_air2[i])
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(air2.index, air2['C%d'%(i+16)], label='air2_C%d'%(i+17) , color=colors_air2[i])
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(air2.index, air2['C%d'%(i+24)], label='air2_C%d'%(i+25) , color=colors_air2[i])
        plt.title('C25-C32')
        plt.legend()

# 显示图形
plt.show()