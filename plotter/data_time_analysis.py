import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

peach_flag = [1,1]
orange_flag = [1,1]
apple_flag = [1,1]
air_flag = [1,1]

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

# peach1_max = peach1.max()
# peach2_max = peach2.max()
# apple1_max = apple1.max()
# apple2_max = apple2.max()
# orange1_max = orange1.max()
# orange2_max = orange2.max()
# air1_max = air1.max()
# air2_max = air2.max()

# # 创建一个新的DataFrame，用于存储两个数据集的均值
# maxs_combined = pd.DataFrame({
#     'peach1': peach1_max,
#     'peach2': peach2_max,
#     'apple1': apple1_max,
#     'apple2': apple2_max,
#     'orange1': orange1_max,
#     'orange2': orange2_max,
#     'air1': air1_max,
#     'air2': air2_max
# })

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
        plt.plot(apple2.index, apple2['C%d'%i], label='apple2_C%d'%(i+1) , color=colors_apple2[i],marker='o', linestyle='-')
        plt.title('C1-C8')
        plt.legend()
        fig2 = plt.figure(num="C9-C16")
        plt.plot(apple2.index, apple2['C%d'%(i+8)], label='apple2_C%d'%(i+9) , color=colors_apple2[i],marker='o', linestyle='-')
        plt.title('C9-C16')
        plt.legend()
        fig3 = plt.figure(num="C17-C24")
        plt.plot(apple2.index, apple2['C%d'%(i+16)], label='apple2_C%d'%(i+17) , color=colors_apple2[i],marker='o', linestyle='-')
        plt.title('C17-C24')
        plt.legend()
        fig4 = plt.figure(num="C25-C32")
        plt.plot(apple2.index, apple2['C%d'%(i+24)], label='apple2_C%d'%(i+25) , color=colors_apple2[i],marker='o', linestyle='-')
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