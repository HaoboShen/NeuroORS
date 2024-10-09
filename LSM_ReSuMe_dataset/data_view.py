
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 初始化图像数据
def init():
    img = torch.zeros(34, 34, 3)
    return [plt.imshow(img)]
 
# 更新图像数据
def update(frame):
    ax.clear()
    ax.axis('off')
    #fig.clf()
    ax.set_title("frame: %d targets: %d"%(frame,targets[(frame-1)//150].item()), fontsize=20)
    img = img_data[frame, :, :, :]
    #print(frame,targets[(frame-1)//150])
    return [plt.imshow(img)]
num = 10
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=150)])

testset = tonic.datasets.NMNIST(save_to='../snntorch-LSM-main-fork/data', transform=frame_transform, train=False)
cached_testset = DiskCachedDataset(testset, cache_path='../snntorch-LSM-main-fork/cache/nmnist/test')
testloader = DataLoader(cached_testset, batch_size=num,shuffle=True)

data, targets = next(iter(testloader))
data_shaped = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4])).type(torch.float32)
# data_shaped = data.type(torch.float32)
fake_channel = torch.zeros(num*150,1,34,34)
print("data_shaped:",data_shaped.shape,"tartgets:",targets)
img_data = torch.cat([data_shaped, fake_channel], dim=1)

# # 使用torch.nonzero()获取非零元素的索引
# non_zero_indices = torch.nonzero(img_data)
# print(non_zero_indices)
# for i in non_zero_indices:
#     if img_data[i[0],i[1],i[2],i[3]] != 1.0 and img_data[i[0],i[1],i[2],i[3]] != 2.0:
#         print(img_data[i[0],i[1],i[2],i[3]])

# # 打印非零元素的值
# for index in non_zero_indices:
#     print(img_data[index])

#提亮图像
img_data = img_data*1.5
img_data = img_data.clamp(0, 1)
img_data = img_data.permute(0, 2, 3, 1)
print("img_data:",img_data.shape)

fig, ax = plt.subplots()

ani = animation.FuncAnimation(fig, update, frames=range(num*150), init_func=init, interval=1)
#ani.save('nminst.gif', writer='pillow', fps=1000)
plt.show()
