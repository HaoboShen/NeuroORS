
# import tonic
# from tonic import DiskCachedDataset
import torch
# import tonic.transforms as transforms
from torch.utils.data import DataLoader
import time
from txt2dataset import txt2Dataset
import torch.utils.data as Data
import numpy as np
from lsm_models import LSM

def targets_to_spike(ts, num_steps, out_sz, active_rate=1.0, inactive_rate=0.0):
    # print("targets_shape",ts.shape)
    spike_train = torch.zeros(ts.shape[0], num_steps, out_sz)
    for i in range(out_sz):
        spike_train[:,:,i] = (torch.rand_like(spike_train[:,:,i]) > 1.0-inactive_rate).float()
    for j in range(ts.shape[0]):
        # print("targets:",ts[j])
        if ts[j] == 0:
            spike_train[j,:,0] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 1:
            spike_train[j,:,1] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 2:
            spike_train[j,:,2] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 3:
            spike_train[j,:,3] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 4.0:
            spike_train[j,:,4] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 5.0:
            spike_train[j,:,5] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 6.0:
            spike_train[j,:,6] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 7.0:
            spike_train[j,:,7] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 8.0:
            spike_train[j,:,8] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 9.0:
            spike_train[j,:,9] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()

    return spike_train

# sensor_size = tonic.datasets.NMNIST.sensor_size
# frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
#                                         transforms.ToFrame(sensor_size=sensor_size, n_time_bins=150)])

# testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)
# cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')
# testloader = DataLoader(cached_testset, batch_size=1)
dataset = txt2Dataset('datasetL5/_100_',num_steps=150)
trainset, testset = Data.random_split(dataset,lengths=[int(0.7 * len(dataset)),len(dataset) - int(0.7 * len(dataset))],generator=torch.Generator().manual_seed(0))
# trainloader = DataLoader(trainset, batch_size=1)
testloader = DataLoader(testset, batch_size=1000,shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
scale = 1
print(device)
data, targets = next(iter(testloader))
# data = torch.rand(10, 150, 2, 34, 34)
print("data:",data.shape)
print("targets:",targets)
data = data.to(device).type(torch.float32)
targets = targets.to(device)/scale
targets_label = targets
targets = targets_to_spike(targets,num_steps=150,out_sz=10).to(device)
alpha = np.float32(np.exp(-1/16))
beta = np.float32(1 - 1/16)

diff_collect = torch.empty(0).to(device)
lsm_net = LSM(1000, 72, 10, np.float32(np.zeros((1000,72))), np.float32(np.zeros((1000,1000))), alpha=alpha, beta=beta, th=20).to(device)
lsm_net.load_state_dict(torch.load("lsm_net_dataset_100_.pth"))

start_time = time.time()
lsm_net.eval()
lsm_net.ReSuMe.train_flag = False
spk_rec = lsm_net(data)
# print("spk_rec",spk_rec)
end_time = time.time()
lsm_out = torch.mean(spk_rec, dim=0)
# print("lsm_out:", lsm_out)
spk_mean = torch.mean(spk_rec,dim=0)
# targets_mean = torch.mean(targets,dim=1)
spk_max,spk_max_idx = torch.max(spk_mean,1)
diff = spk_max_idx - targets_label
non_zero = torch.nonzero(diff)
# spk_result_one_hot = torch.zeros_like(spk_mean)
# spk_result_one_hot[spk_max_idx] = 1
print("spk_max:",spk_max,"spk_max_idx:",spk_max_idx)
print("diff:",diff)
print("percentage of correct",1.0 - non_zero.shape[0]/diff.shape[0])
