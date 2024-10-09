import numpy as np
import matplotlib.pyplot as plt
import pandas

# 读取csv
df = pandas.read_csv('peach2_diff.csv')
# 得到信号长度
N = df.shape[0]

# 定义信号
sampling_rate = 1  # 采样频率，单位：Hz
duration = N  # 信号时长，单位：秒
t = np.linspace(0, duration, sampling_rate * duration, endpoint=False)  # 时间向量
print("t:",t)
signal = df['C8'].values  # 信号

# 把空值替换为0
signal = np.nan_to_num(signal)
 
# 计算FFT
fft_result = np.fft.fft(signal)
# for r in fft_result:
#     if r!=np.nan:
#         print(r)
# print("fft_result:",fft_result)

# 计算频率轴
n = signal.size
sample_frequency = 1 / (t[1] - t[0])
freq = np.fft.fftfreq(n, d=sample_frequency)

# 取FFT结果的一半（Nyquist定理）
half_n = n // 2
fft_result = fft_result[:half_n]
freq = freq[:half_n]

# 取绝对值获取振幅谱
fft_magnitude = np.abs(fft_result)

# 绘制频谱图
plt.plot(freq, fft_magnitude)
plt.title("FFT Magnitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()