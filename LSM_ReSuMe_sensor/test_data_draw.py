import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_prepare_data(file_path, column_count=24):
    data = pd.read_csv(file_path)
    data = pd.DataFrame(np.nan_to_num(data.values), columns=['C%d' % i for i in range(column_count)] + ['result'])
    return data

def plot_combined_data_with_background(data, title, colors, labels, min_span_width=10, font_name='Times New Roman', font_size=20):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(data.shape[1] - 1):
        ax.plot(data.index, data[f'C{i}'], label=f'C{i+1}')

    # 画虚线和标注
    annotations = {
        24: 'peach placed',
        74: 'peach removed',
        145: 'orange placed',
        215: 'orange removed'
    }

    annotation_positions = sorted(annotations.keys())
    next_region_starts = []

    # 避免标注与标题冲突，调整标注位置
    ylim = ax.get_ylim()
    y_offset = (ylim[1] - ylim[0]) * 0.1
    annotation_y = ylim[1] - y_offset * 0.3  # 提高虚线标志位置

    for pos, text in annotations.items():
        ax.axvline(x=pos, color='k', linestyle='--', alpha=0.5)
        ax.text(pos, annotation_y, text, ha='center', va='top', color='black', fontdict={'family': font_name, 'size': font_size - 4}, clip_on=False)

    pre_result = data['result'][0]
    pre_index = 0
    significant_starts = []

    for index, result in enumerate(data['result']):
        if index == 0:
            continue
        if result != pre_result:
            start_time = pre_index
            end_time = index
            span_width = end_time - start_time
            color = colors.get(int(pre_result), 'none')
            ax.axvspan(start_time, end_time, facecolor=color, edgecolor='k', alpha=0.5, zorder=-1)
            if span_width >= min_span_width:
                significant_starts.append(start_time)
                mid_time = (start_time + end_time) / 2
                label_text = labels.get(int(pre_result), f'Class {int(pre_result)}')
                ax.text(mid_time, ylim[1] - y_offset * 2.5, label_text, ha='center', va='center', color='black',
                        fontdict={'family': font_name, 'size': font_size + 4})
            pre_index = index
        pre_result = result

    # 最后一个区间
    start_time = pre_index
    end_time = len(data)
    span_width = end_time - start_time
    color = colors.get(int(pre_result), 'none')
    ax.axvspan(start_time, end_time, facecolor=color, edgecolor='k', alpha=0.5, zorder=-1)
    if span_width >= min_span_width:
        significant_starts.append(start_time)
        mid_time = (start_time + end_time) / 2
        label_text = labels.get(int(pre_result), f'Class {int(pre_result)}')
        ax.text(mid_time, ylim[1] - y_offset * 2.5, label_text, ha='center', va='center', color='black',
                fontdict={'family': font_name, 'size': font_size + 4})

    # 标注距离并用双箭头表示
    arrow_y_pos = annotation_y - y_offset * 1.2
    for pos in annotation_positions:
        next_start = None
        for start in significant_starts:
            if start > pos:
                next_start = start
                break
        if next_start:
            mid_pos = (pos + next_start) / 2
            distance = next_start - pos
            ax.annotate(
                '',
                xy=(pos, arrow_y_pos),
                xytext=(next_start, arrow_y_pos),
                arrowprops=dict(arrowstyle='<->', color='black')
            )
            ax.text(mid_pos, arrow_y_pos - y_offset * 0.5, f'{distance}', ha='center', va='center', color='black', fontdict={'family': font_name, 'size': font_size}, clip_on=False)

    ax.set_title(title, fontdict={'family': font_name, 'size': font_size + 4}, pad=10)  # 紧贴标题和图片

    # Set axis labels and ticks font size
    ax.set_xlabel('Time(S)', fontdict={'family': font_name, 'size': font_size + 4})
    ax.set_ylabel('Voltage(V)', fontdict={'family': font_name, 'size': font_size + 4})
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    # Set tick labels font family to Times New Roman
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(font_name)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 紧贴布局
    plt.savefig(f'{title}.png')

def main():
    colors = {0: 'lightgreen', 1: 'lightcoral', 2: 'thistle'}  # 自定义背景颜色
    labels = {0: 'air', 1: 'orange', 2: 'peach'}  # 自定义类别对应的文字
    file_paths = {
        'peach_orange_recognition_test': 'peach_orange_test_data.csv'  # 修改文件路径
    }

    for key in file_paths.keys():
        data = load_and_prepare_data(file_paths[key])
        title = key.replace('_', ' ').title()
        plot_combined_data_with_background(data, title, colors, labels, font_size=20)  # 设置字体大小

    plt.show()

if __name__ == '__main__':
    main()
