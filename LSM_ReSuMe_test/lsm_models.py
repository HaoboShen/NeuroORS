import torch
# import torchvision
# from torch.utils.data import DataLoader
import torch.nn as nn
# import time
import snntorch as snn
from ReSuMe_models import *

class LSM(nn.Module):

    def __init__(self, N, in_sz, out_sz, Win, Wlsm, alpha, beta, th ,train_flag=False):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_sz, N)
        self.fc1.weight = nn.Parameter(torch.from_numpy(Win))
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th)
        self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(Wlsm))
        # self.lsm_out_syn = nn.Linear(N, N)
        self.train_flag = train_flag
        # self.lsm_out_syn.weight = nn.Parameter(torch.ones(out_sz, N))
        self.ReSuMe = ReSuMe(N=N,out_sz=out_sz ,alpha=alpha, beta=beta, input=self.lsm, train_flag=self.train_flag)
        # self.lsm_out_syn = nn.Linear(N, 10)
        # self.lsm_out_syn.weight = nn.Parameter(torch.ones(N, 10))

        # #ReSuMe learning rule
        # self.teach_in_syn = nn.Linear(10, 1)
        # self.teach_in_syn.weight = nn.Parameter(torch.ones(10, 1))
        # self.teach = snn.Synaptic(alpha=alpha,beta=beta)
        # # self.teach_out_syn = nn.Linear(10, N)
        # # self.teach_out_syn.weight = 1.0
        # self.readout = snn.Synaptic(alpha=alpha,beta=beta)
        # self.anti_stdp = STDPLearner(self.readout, self.lsm,2,2)
        # self.stdp = STDPLearner(self.lsm_out_syn, self.teach,2,2)

    def forward(self, x):

        #print(x.size())
        num_steps = x.size(1)
        print("num_steps: ", num_steps)
        spk, syn, mem = self.lsm.init_rsynaptic()
        # spk_rec = []
        for step in range(num_steps):
            curr = self.fc1(self.flatten(x[:,step,:,:,:]))
            spk, syn, mem = self.lsm(curr, spk, syn, mem)

            # print("spk:",spk)
            # print("mon:",self.mon.records.pop())
            nonzero_indices = torch.nonzero(spk, as_tuple=False)
            print("nonzero_indices: ", nonzero_indices)
            # out = self.ReSuMe(spk)
            out = spk
            return out
            # if self.train_flag:
            #     output = self.ReSuMe(curr_out)
            #     delta_w = output[0]
            #     out = output[1]
            #     print(delta_w.shape)
            #     print(self.lsm_out_syn.weight.shape)
            #     self.lsm_out_syn.weight = nn.Parameter(delta_w+self.lsm_out_syn.weight)
            #     print("weight:",self.lsm_out_syn.weight.shape)
            #     print("out",out.shape)
            #     return out
            # else:
            #     out = self.ReSuMe(curr_out)
            #     return out
            # print("stdp",self.stdp.step())
            # print("anti-stdp",self.anti_stdp.step())
            # print("stdp+anti-stdp",self.stdp.step()+self.anti_stdp.step())
            # spk_rec.append(spk)

        # spk_rec_out = torch.stack(spk_rec)
        # return spk_rec_out

