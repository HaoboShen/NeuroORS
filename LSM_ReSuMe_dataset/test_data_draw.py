# test_data_draw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

alcohol_vinegar_flag = [1,1]
apple_peach_orange_flag = [1,1]

alcohol_vinegar_oneshot = pd.read_csv(r'test_alcochol_vinegar_oneshot.csv')
alcohol_vinegar_repeat = pd.read_csv(r'test_alcochol_vinegar_repeat.csv')
apple_peach_orange_oneshot = pd.read_csv(r'test_apple_peach_orange_oneshot.csv')
apple_peach_orange_repeat = pd.read_csv(r'test_apple_peach_orange_repeat.csv')
alcohol_vinegar_oneshot = pd.DataFrame(np.nan_to_num(alcohol_vinegar_oneshot.values),columns = ['C%d'%i for i in range(24)]+['result'])
alcohol_vinegar_repeat = pd.DataFrame(np.nan_to_num(alcohol_vinegar_repeat.values),columns = ['C%d'%i for i in range(24)]+['result'])
apple_peach_orange_oneshot = pd.DataFrame(np.nan_to_num(apple_peach_orange_oneshot.values),columns = ['C%d'%i for i in range(24)]+['result'])
apple_peach_orange_repeat = pd.DataFrame(np.nan_to_num(apple_peach_orange_repeat.values),columns = ['C%d'%i for i in range(24)]+['result'])

colors = ['blue','red','green','yellow']
# 为每个列绘制图形
if alcohol_vinegar_flag[0]:
    for i in range(8):
        fig1 = plt.figure(num="alcohol_vinegar_oneshot_C1-C8")
        plt.plot(alcohol_vinegar_oneshot.index, alcohol_vinegar_oneshot['C%d'%i], label='C%d'%(i+1))
        for index,result in enumerate(alcohol_vinegar_oneshot['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
            
                # 计算区间的开始和结束时间
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C1-C8')
        plt.legend()

        fig2 = plt.figure(num="alcohol_vinegar_oneshot_C9-C16")
        plt.plot(alcohol_vinegar_oneshot.index, alcohol_vinegar_oneshot['C%d'%(i+8)], label='C%d'%(i+9))
        for index,result in enumerate(alcohol_vinegar_oneshot['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
                # 计算区间的开始和结束时间
            
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C9-C16')
        plt.legend()

        fig3 = plt.figure(num="alcohol_vinegar_oneshot_C17-C24")
        plt.plot(alcohol_vinegar_oneshot.index, alcohol_vinegar_oneshot['C%d'%(i+16)], label='C%d'%(i+17))
        for index,result in enumerate(alcohol_vinegar_oneshot['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
            
                # 计算区间的开始和结束时间
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C17-C24')
        plt.legend()
if alcohol_vinegar_flag[1]:
    for i in range(8):
        fig4 = plt.figure(num="alcohol_vinegar_repeat_C1-C8")
        plt.plot(alcohol_vinegar_repeat.index, alcohol_vinegar_repeat['C%d'%i], label='C%d'%(i+1))
        for index,result in enumerate(alcohol_vinegar_repeat['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
                # 计算区间的开始和结束时间
            
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C1-C8')
        plt.legend()

        fig5 = plt.figure(num="alcohol_vinegar_repeat_C9-C16")
        plt.plot(alcohol_vinegar_repeat.index, alcohol_vinegar_repeat['C%d'%(i+8)], label='C%d'%(i+9))
        for index,result in enumerate(alcohol_vinegar_repeat['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
                # 计算区间的开始和结束时间
            
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C9-C16')
        plt.legend()

        fig6 = plt.figure(num="alcohol_vinegar_repeat_C17-C24")
        plt.plot(alcohol_vinegar_repeat.index, alcohol_vinegar_repeat['C%d'%(i+16)], label='C%d'%(i+17))
        for index,result in enumerate(alcohol_vinegar_repeat['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue
            if result != pre_result:
            
                start_time = pre_index
                end_time = index
                pre_index = index
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C17-C24')
        plt.legend()
if apple_peach_orange_flag[0]:
    for i in range(8):
        fig7 = plt.figure(num="apple_peach_orange_oneshot_C1-C8")
        plt.plot(apple_peach_orange_oneshot.index, apple_peach_orange_oneshot['C%d'%i], label='C%d'%(i+1))
        for index,result in enumerate(apple_peach_orange_oneshot['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
                # 计算区间的开始和结束时间
            
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C1-C8')
        plt.legend()

        fig8 = plt.figure(num="apple_peach_orange_oneshot_C9-C16")
        plt.plot(apple_peach_orange_oneshot.index, apple_peach_orange_oneshot['C%d'%(i+8)], label='C%d'%(i+9))
        for index,result in enumerate(apple_peach_orange_oneshot['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
                # 计算区间的开始和结束时间
            
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C9-C16')
        plt.legend()
        fig9 = plt.figure(num="apple_peach_orange_oneshot_C17-C24")
        plt.plot(apple_peach_orange_oneshot.index, apple_peach_orange_oneshot['C%d'%(i+16)], label='C%d'%(i+17))
        for index,result in enumerate(apple_peach_orange_oneshot['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue
            if result != pre_result:
            
                start_time = pre_index
                end_time = index
                pre_index = index
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C17-C24')
        plt.legend()
if apple_peach_orange_flag[1]:
    for i in range(8):
        fig10 = plt.figure(num="apple_peach_orange_repeat_C1-C8")
        plt.plot(apple_peach_orange_repeat.index, apple_peach_orange_repeat['C%d'%i], label='C%d'%(i+1))
        for index,result in enumerate(apple_peach_orange_repeat['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
                # 计算区间的开始和结束时间
                
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C1-C8')
        plt.legend()

        fig11 = plt.figure(num="apple_peach_orange_repeat_C9-C16")
        plt.plot(apple_peach_orange_repeat.index, apple_peach_orange_repeat['C%d'%(i+8)], label='C%d'%(i+9))
        for index,result in enumerate(apple_peach_orange_repeat['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue  # 跳过第一个数据点，因为没有前一个点来定义区间的开始
            if result != pre_result:  # 如果当前分类与前一个分类不同
                
                # 计算区间的开始和结束时间
                start_time = pre_index
                end_time = index
                pre_index = index

                # 使用axvspan绘制背景颜色
                # print(start_time,end_time,pre_result,colors[int(pre_result)])
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C9-C16')
        plt.legend()

        fig12 = plt.figure(num="apple_peach_orange_repeat_C17-C24")
        plt.plot(apple_peach_orange_repeat.index, apple_peach_orange_repeat['C%d'%(i+16)], label='C%d'%(i+17))
        for index,result in enumerate(apple_peach_orange_repeat['result']):
            if index == 0:
                pre_result = result
                pre_index = index
                continue
            if result != pre_result:
                
                start_time = pre_index
                end_time = index
                pre_index = index
                plt.axvspan(start_time, end_time, color=colors[int(pre_result)], alpha=0.01)
            pre_result = result

        plt.title('C17-C24')
        plt.legend()
# save figures
fig1.savefig('alcohol_vinegar_oneshot_C1-C8.png')
fig2.savefig('alcohol_vinegar_oneshot_C9-C16.png')
fig3.savefig('alcohol_vinegar_oneshot_C17-C24.png')
fig4.savefig('alcohol_vinegar_repeat_C1-C8.png')
fig5.savefig('alcohol_vinegar_repeat_C9-C16.png')
fig6.savefig('alcohol_vinegar_repeat_C17-C24.png')
fig7.savefig('apple_peach_orange_oneshot_C1-C8.png')
fig8.savefig('apple_peach_orange_oneshot_C9-C16.png')
fig9.savefig('apple_peach_orange_oneshot_C17-C24.png')
fig10.savefig('apple_peach_orange_repeat_C1-C8.png')
fig11.savefig('apple_peach_orange_repeat_C9-C16.png')
fig12.savefig('apple_peach_orange_repeat_C17-C24.png')

plt.show()
