# import tonic
# from tonic import DiskCachedDataset
# import tonic.transforms as transforms
from matplotlib import pyplot as plt
import torch 
import torch.utils.data as Data
# from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import numpy as np
import time

from lsm_weight_definitions import *
from lsm_models import LSM
from ReSuMe_models import *

from csv2dataset import CSV2Dataset


def targets_to_spike(ts, num_steps, out_sz, active_rate=1.0, inactive_rate=0.0):
    # print("targets_shape",ts.shape)
    spike_train = torch.zeros(ts.shape[0], num_steps, out_sz)
    for i in range(out_sz):
        spike_train[:,:,i] = (torch.rand_like(spike_train[:,:,i]) > 1.0-inactive_rate).float()
    for j in range(ts.shape[0]):
        # print("targets:",ts[j])
        if ts[j] == 0.0:
            spike_train[j,:,0] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 1.0:
            spike_train[j,:,1] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 2.0:
            spike_train[j,:,2] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 3.0:
            spike_train[j,:,3] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
    # print("spike_train:",spike_train)
    return spike_train

parser = argparse.ArgumentParser(description='Train an LSM')

parser.add_argument('--time_steps', default=10, type=int, help='number of time steps in model')
parser.add_argument('--batch_size', default=25, type=int, help='batch size for training')
parser.add_argument('--tau_v', default=16.0, type=float, help='membrane time constant')
parser.add_argument('--tau_i', default=16.0, type=float, help='synaptic time constant')
parser.add_argument('--th', default=20, type=float, help='threshold for spiking')
parser.add_argument('--LqWin', default=27, type=float, help='LqWin')
parser.add_argument('--LqWlsm', default=2, type=float, help='LqWlsm')
parser.add_argument('--in_conn_density', default=0.15, type=float, help='in_conn_density')
parser.add_argument('--lam', default=9, type=float, help='d')
parser.add_argument('--inh_fr', default=0.2, type=float, help='inhibitory firing rate')
parser.add_argument('--Nx', default=10, type=int, help='Nx')
parser.add_argument('--Ny', default=10, type=int, help='Ny')
parser.add_argument('--Nz', default=10, type=int, help='Nz')

args = parser.parse_args()

def main():

    # sensor_size = tonic.datasets.NMNIST.sensor_size
    # frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
    #                                       transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.time_steps)])

    # trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    # testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

    # cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    # cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

    
    # frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=(24,1), n_time_bins=args.time_steps)])
    # transform = transforms.Compose([transforms.ToTensor()])
    dataset = CSV2Dataset('orange_peach_diff/',transform=None,num_steps=args.time_steps)
    trainset, testset = Data.random_split(dataset,lengths=[int(0.9 * len(dataset)),len(dataset) - int(0.9 * len(dataset))],generator=torch.Generator().manual_seed(0))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    model_name = 'lsm_net_orange_peach_diff.pth'
    data, targets = next(iter(trainloader))
    # print("data", data.shape)
    # print("targets", targets.shape)
    # flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1))
    # print("flat_data", flat_data.shape)


    in_sz = data.shape[-1]
    out_sz = 3

    #Set neuron parameters
    curr_prefac = np.float32(1/args.tau_i)
    alpha = np.float32(np.exp(-1/args.tau_i))
    beta = np.float32(1 - 1/args.tau_v)

    Win, Wlsm = initWeights1(LqWin=args.LqWin, LqWlsm=args.LqWlsm, in_conn_density=args.in_conn_density, in_size=in_sz, lam=args.lam, 
                             inh_fr=args.inh_fr, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, init_Wlsm=True, W_lsm=None)

    N = Wlsm.shape[0]
    # save win and Wlsm
    np.save('Win.npy', Win)
    np.save('Wlsm.npy', Wlsm)
    # load win and Wlsm
    # Win = np.load('Win.npy')
    # Wlsm = np.load('Wlsm.npy')
    # scale = 1
    scale = 10000
    lsm_net = LSM(N, in_sz, out_sz, np.float32(curr_prefac*Win), np.float32(curr_prefac*Wlsm), alpha=alpha, beta=beta, th=args.th).to(device)
    lsm_net.eval()
    #Run with no_grad for LSM
    for i in range(5):
        with torch.no_grad():
            start_time = time.time()

            # initialize empty tensors
            # in_train = torch.empty(0).to(device)

            for i, (data, targets) in enumerate(iter(trainloader)):

                if i%25 == 24:
                    print("train batches completed: ", i)
                

                data = data.to(device).type(torch.float32)
                targets = targets/scale
                targets = targets_to_spike(targets,num_steps=args.time_steps,out_sz=out_sz).to(device)
                
                # print("data:",data)
                
                # flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device).type(torch.float32)


                # print("init_done")
                
                lsm_net.teach_input = targets
                # print("targets:",targets)
                # print("targets_shape:",targets.shape)
                lsm_net.ReSuMe.train_flag = True
                # print("data",data)
                spk_rec = lsm_net(data)
                # print("spk_rec", spk_rec)
                print("weight:",lsm_net.ReSuMe.readout.readout_in_syn.weight)
                # weight.append(lsm_net.ReSuMe.readout.readout_in_syn.weight)
                # for i in range(lsm_net.ReSuMe.readout.readout_in_syn.weight.shape[0]):
                #     for j in range(lsm_net.ReSuMe.readout.readout_in_syn.weight.shape[1]):
                #         if lsm_net.ReSuMe.readout.readout_in_syn.weight[i,j] > 0.0:
                #             print("weight:",lsm_net.ReSuMe.readout.readout_in_syn.weight[i,j])

                print(i)
        # weight = torch.stack(weight)
        # weight = weight.to('cpu')
        end_time = time.time()


        print("running time of training epoch: ", end_time - start_time, "seconds")

        # initialize empty tensors
        # in_test = torch.empty(0).to(device)
        # lsm_out_test = torch.empty(0).to(device)
        # lsm_label_test = torch.empty(0).to(device)
        diff_collect = torch.empty(0).to(device)
        for i, (data, targets) in enumerate(iter(testloader)):

            if i%25 == 24:
                print("test batches completed: ", i)

            data = data.to(device).type(torch.float32)
            targets = targets.to(device)
            
            targets = targets/scale
            targets_label = targets
            print("test targets:",targets)
            targets = targets_to_spike(targets,num_steps=args.time_steps,out_sz=out_sz).to(device)
            # print("test targets one hot:",targets)
            print("weight_mean:",torch.mean(lsm_net.ReSuMe.readout.readout_in_syn.weight))


            # print("init_done")
            lsm_net.eval()
            lsm_net.ReSuMe.train_flag = False
            spk_rec = lsm_net(data)
            spk_mean = torch.mean(spk_rec,dim=0)
            spk_max,spk_max_idx = torch.max(spk_mean,1)
            diff = spk_max_idx - targets_label
            print("spk:",spk_mean)
            print("spk_max:",spk_max,"spk_max_idx:",spk_max_idx)
            print("diff:",diff)
            diff_collect = torch.cat((diff_collect,diff),dim=0)
        non_zero = torch.nonzero(diff_collect)
        print("diff_collect_shape:",diff_collect.shape)
        print("diff_collect:",diff_collect)
        print("non_zero:",non_zero.shape)
        # print("non_zero:",non_zero)
        print("percentage of correct:",1.0 - non_zero.shape[0]/diff_collect.shape[0])
        torch.save(lsm_net.state_dict(), model_name)
            # print("spk_rec:",spk_rec)
            # print("spk_rec_mean:",spk_mean)
            # print("spk_result_one_hot:",spk_result_one_hot)

            # lsm_out = torch.mean(spk_rec, dim=0)
            # print("spk_rec", spk_rec.shape)
            # # lsm_out = spk_rec

            # print("flat_data", flat_data.type())
            # in_test = torch.cat((in_test, torch.mean(flat_data, dim=0)), dim=0)
            # lsm_out_test = torch.cat((lsm_out_test, lsm_out), dim=0)
            # lsm_label_test = torch.cat((lsm_label_test, targets), dim=0)
            # print("lsm_out_test", lsm_out_test.shape)
    # cmap = plt.get_cmap('tab10')
    
    # print("weight_shape:",weight.shape)
    # w = weight.reshape(-1,weight.shape[-1])
    # t_weight = torch.arange(0,w.shape[0]).to('cpu')
    # plt.plot(t_weight, w[0], c=cmap(4))
    # plt.xlim(-0.5, weight.shape[0] + 0.5)
    # plt.ylabel('$weight$', rotation=0)
    # plt.yticks([w.min().item(), w.max().item()])
    # plt.xlabel('time-step')
    # plt.show()
    # print(lsm_out_train.shape)
    # print(lsm_out_test.shape)

    # print(in_train.shape)
    # print(in_test.shape)

    # print("mean in spiking (train) : ", torch.mean(in_train))
    # print("mean in spiking (test) : ", torch.mean(in_test))

    # print("mean LSM spiking (train) : ", torch.mean(lsm_out_train))
    # print("mean LSM spiking (test) : ", torch.mean(lsm_out_test))

    # print("training linear model:")

    # # to numpy
    # lsm_out_train_np = lsm_out_train.cpu().numpy()
    # lsm_out_test_np = lsm_out_test.cpu().numpy()
    # lsm_label_train_np = lsm_label_train.cpu().numpy()
    # lsm_label_test_np = lsm_label_test.cpu().numpy()

    # print("lsm_out_train_np:",lsm_out_train_np.shape)
    # print("lsm_label_train_np:",lsm_label_train_np.shape)
    # plt.plot(lsm_out_train_np.T)
    # clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    # clf.fit(lsm_out_train_np, lsm_label_train_np)

    # score = clf.score(lsm_out_test_np, lsm_label_test_np)
    # print("test score = " + str(score))

    # clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    # clf.fit(in_train, lsm_label_train)

    # score = clf.score(in_test, lsm_label_test)
    # print("test score = " + str(score))
    # torch.save(lsm_net, 'lsm_net.pth')

    # with open('clf.pickle', 'wb') as f:
    #     pickle.dump(clf, f)



if '__main__' == __name__:
    main()