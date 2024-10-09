from typing import Callable, Union
import torch
# import torchvision
# from torch.utils.data import DataLoader
import torch.nn as nn
# import time
import snntorch as snn
# from stdp_learner import *  
from snntorch.functional import probe

    
class readout(nn.Module):
    def __init__(self,N,out_sz,alpha,beta):
        super().__init__()
        self.readout_in_syn = nn.Linear(N, out_sz,bias=False)
        self.readout_in_syn.weight = nn.Parameter(torch.ones(out_sz, N)*-1.0)
        # self.readout_in_syn.bias = nn.Parameter(torch.zeros(out_sz))
        self.readout = snn.Synaptic(alpha=alpha,beta=beta)
        self.mask = 1.0
        self.spk = None
        self.syn, self.mem = self.readout.init_synaptic()

    def forward(self,x):
        curr1=self.readout_in_syn(x)
        self.spk,self.syn,self.mem=self.readout(curr1 , self.syn, self.mem)
        return self.spk
# half stdp
class STDP(nn.Module):
    def __init__(
        self,
        synapse, sn,
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super().__init__()
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = probe.InputMonitor(synapse)
        self.out_spike = None
        if sn is None:
            self.out_spike_monitor = None
        else:
            self.out_spike_monitor = probe.OutputMonitor(sn)
        print("synapse:",synapse)
        print("sn:",sn)
        self.trace_pre = None
        self.trace_post = None

    def reset(self):
        # super(STDP, self).reset()
        self.trace_pre = None
        self.trace_post = None
        self.in_spike_monitor.clear_recorded_data()
        if self.out_spike_monitor is not None:
            self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def stdp_linear_single_step(self,
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
        if trace_pre is None:
            trace_pre = 0.

        if trace_post is None:
            trace_post = 0.

        trace_pre = trace_pre - trace_pre / tau_pre + in_spike  # shape = [batch_size, N_in]

        if isinstance(out_spike, tuple):
            delta_w_post = (trace_pre.unsqueeze(1) * out_spike[0].unsqueeze(2)).sum(0)
        if isinstance(out_spike, torch.Tensor):
            delta_w_post = (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
        # print("delta_w_pre:",delta_w_pre)
        # print("delta_w_post:",delta_w_post)
        return trace_pre, trace_post, delta_w_post


    def forward(self, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)  # [batch_size, N_in]
            if self.out_spike_monitor is None:
                out_spike = self.out_spike
            else:
                out_spike = self.out_spike_monitor.records.pop(0)  # [batch_size, N_out]
            self.trace_pre, self.trace_post, dw = self.stdp_linear_single_step(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post,
                self.tau_pre, self.tau_post,
                self.f_pre, self.f_post
            )
            if scale != 1.:
                dw *= scale
            delta_w = dw if (delta_w is None) else (delta_w + dw)

            return delta_w#ReSuMe learning rule
        
class anti_STDP(nn.Module):
    def __init__(
        self,
        synapse, sn,
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super().__init__()
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = probe.InputMonitor(synapse)
        self.out_spike_monitor = probe.OutputMonitor(sn)
        print("synapse:",synapse)
        print("sn:",sn)
        self.trace_pre = None
        self.trace_post = None

    def reset(self):
        # super(anti_STDP, self).reset()
        self.trace_pre = None
        self.trace_post = None
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def stdp_linear_single_step(self,
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
        if trace_pre is None:
            trace_pre = 0.

        if trace_post is None:
            trace_post = 0.

        trace_pre = trace_pre - trace_pre / tau_pre + in_spike  # shape = [batch_size, N_in]
        delta_w_post = -(trace_pre.unsqueeze(1) * out_spike[0].unsqueeze(2)).sum(0)

        return trace_pre, trace_post, delta_w_post


    def forward(self, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)  # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)  # [batch_size, N_out]


            self.trace_pre, self.trace_post, dw = self.stdp_linear_single_step(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post,
                self.tau_pre, self.tau_post,
                self.f_pre, self.f_post
            )
            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

            return delta_w#ReSuMe learning rule
        
class ReSuMe(nn.Module):
    def __init__(self, N, out_sz, alpha, beta):
        super().__init__()
        self.train_flag = False
        self.readout = readout(N=N,out_sz=out_sz,alpha=alpha,beta=beta)
        self.teach_input = None
        self.anti_stdp_learn = anti_STDP(self.readout.readout_in_syn, self.readout.readout,2,2,f_pre=self.f_weight, f_post=self.f_weight)
        self.stdp_learn = STDP(self.readout.readout_in_syn, self.teach_input,2,2,f_pre=self.f_weight, f_post=self.f_weight)

    def f_weight(self,x):
        return torch.clamp(x, -1, 1.)
    
    def forward(self,x):

        if self.train_flag:
            # self.teach(self.teach_input)
            self.stdp_learn.out_spike = self.teach_input
            # self.readout.mask = self.teach_input
            # print("teach:",self.teach_input)
            out = self.readout(x)
            # w1 = self.stdp_learn(scale=0.001)
            w1 = self.stdp_learn(scale=0.001)
            # print("w1:",w1)
            # w1_non_zero_values = w1[w1!=0]
            # print("w1_non_zero:",w1_non_zero_values.mean().item())
            # print("w1:",w1)
            # w2 = self.anti_stdp_learn(scale=0.001)
            w2 = self.anti_stdp_learn(scale=0.001)
            # print("w2:",w2)
            # w2_non_zero_values = w2[w2!=0]
            # print("w2_non_zero:",w2_non_zero_values.mean().item())
            # print("w1+w2:",(w1+w2).mean().item())
            # print("w2:",w2.shape)
            self.readout.readout_in_syn.weight = nn.Parameter(w1+w2+self.readout.readout_in_syn.weight)
            # print("weight:",self.readout.readout_in_syn.weight)
            # print("teach_input:",self.teach_input)
            # print("w1:",w1)
            # print("w2:",w2)
            # self.readout.readout_in_syn.weight = nn.Parameter(torch.clamp(w1+w2+self.readout.readout_in_syn.weight,-1,1))
            # self.readout.mask = 1.0
            # self.readout.readout_in_syn.weight = nn.Parameter(w1+w2.T+self.readout.readout_in_syn.weight)
            # print("w:",self.readout.readout_in_syn.weight.shape)
        else:
            out = self.readout(x)

        return out
