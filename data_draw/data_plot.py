import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 设置全局字体为 Times New Roman
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 20})
# read data
orange_A = pd.read_csv('E-nose data/orange_A_14400.csv')
orange_A = pd.DataFrame(np.nan_to_num(orange_A.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
orange_B = pd.read_csv('E-nose data/orange_B_14400.csv')
orange_B = pd.DataFrame(np.nan_to_num(orange_B.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
peach_A = pd.read_csv('E-nose data/peach_A_14400.csv')
peach_A = pd.DataFrame(np.nan_to_num(peach_A.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
peach_B = pd.read_csv('E-nose data/peach_B_14400.csv')
peach_B = pd.DataFrame(np.nan_to_num(peach_B.values),columns = ['f%d'%i for i in range(2)]+['t','h']+['C%d'%i for i in range(32)]+['u%d'%i for i in range(4)])
print("orange_A:",orange_A)
print("orange_B:",orange_B)
print("peach_A:",peach_A)
print("peach_B:",peach_B)

# plot data
plt.figure(figsize=(10, 6))

# for i in range(24):
#     plt.plot(range(orange_A.shape[0]), orange_A['C%d'%i], label='C%d'%i)
# plt.title('Orange',fontsize=24)

for i in range(24):
    plt.plot(range(peach_A.shape[0]), peach_A['C%d'%i], label='C%d'%i)
plt.title('Peach',fontsize=24)

plt.xlabel('Time(S)', fontsize=24)
plt.ylabel('Voltage(V)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# # 去掉图像边距
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, hspace=0.2)
plt.savefig('Peach.png')
# plt.savefig('Orange.png')
plt.show()

