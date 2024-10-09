import torch
# import torchvision
# from torch.utils.data import DataLoader
import torch.nn as nn
# import time
import snntorch as snn
from ReSuMe_models import *

class LSM(nn.Module):

    def __init__(self, N, in_sz, out_sz, Win, Wlsm, alpha, beta, th):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_sz, N)
        self.fc1.weight = nn.Parameter(torch.from_numpy(Win))
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th)
        self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(Wlsm))
        self.teach_input = None
        self.ReSuMe = ReSuMe(N=N,out_sz=out_sz ,alpha=alpha, beta=beta)
        self.spk, self.syn, self.mem = self.lsm.init_rsynaptic()
        

    def forward(self, x):
        
        #print(x.size())
        num_steps = x.size(1)
        # print("num_steps: ", num_steps)
        self.spk, self.syn, self.mem = self.lsm.init_rsynaptic()
        
        # readout init
        self.ReSuMe.readout.syn,self.ReSuMe.readout.mem = self.ReSuMe.readout.readout.init_synaptic()
        self.ReSuMe.readout.spk = None
        self.ReSuMe.readout.mask = 1.0

        # # teach init
        # self.ReSuMe.teach.syn,self.ReSuMe.teach.mem = self.ReSuMe.teach.teach.init_synaptic()
        # self.ReSuMe.teach.spk = None

        # stdp and anti-stdp reset
        self.ReSuMe.anti_stdp_learn.reset()
        self.ReSuMe.stdp_learn.reset()

        # spk_rec = []
        out_collect = []
        for step in range(num_steps):
            # start_time = time.time()
            curr = self.fc1(self.flatten(x[:,step,:,:]))
            print("curr:",curr)
            # curr_time = time.time()
            self.spk, self.syn, self.mem = self.lsm(curr, self.spk, self.syn, self.mem)
            # non_zero = torch.nonzero(self.spk)
            # print("non_zeros:",non_zero.shape)
            if self.teach_input is not None:
                self.ReSuMe.teach_input = self.teach_input[:,step,:]
            # print("teach_input:",self.ReSuMe.teach_input)
            # print("teach_spk:",self.ReSuMe.teach.spk)
            out = self.ReSuMe(self.spk)
            out_collect.append(out)
        
        out_collect = torch.stack(out_collect)
        return out_collect
        #     print("spk: ", spk.size())
        #     # spk_time = time.time()
        #     spk_rec.append(spk)
        #     # end_time = time.time()
            
        #     #print(abs(mem).max())
        # spk_rec_out = torch.stack(spk_rec)
        # #print(spk_rec_out.size())
        # # print("running time of fc1: ", curr_time - start_time, "seconds")
        # # print("running time of spk: ", spk_time - curr_time, "seconds")
        # # print("running time of append: ", end_time - spk_time, "seconds")
        # # print("running time of LSM forward: ", end_time - start_time, "seconds")
        # return spk_rec_out
