from stdp_learner import *
import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np

def f_weight(x):
    return torch.clamp(x, -1, 1.)

tau_pre = 2.
tau_post = 2.
T = 128
N = 1
lr = 0.01
alpha = np.float32(np.exp(-1/16.0))
beta = np.float32(1 - 1/16.0)
class net(nn.Module):
    def __init__(self,N,out_sz,alpha,beta):
        super().__init__()
        self.readout_in_syn = nn.Linear(N, out_sz,bias=False)
        self.readout_in_syn.weight = nn.Parameter(torch.ones(N, out_sz)*0.05)
        # self.readout_in_syn.bias = nn.Parameter(torch.ones(out_sz)*0.001)
        self.readout = snn.Synaptic(alpha=alpha,beta=beta)
        self.spk = None
        self.syn, self.mem = self.readout.init_synaptic()
        self.syn = torch.zeros(1,1).to(device)
        self.mem = torch.zeros(1,1).to(device)
        # print("self.mem_init",self.mem,"self.syn_init",self.syn,"self.spk_init",self.spk)
    def forward(self,x):
        curr1=self.readout_in_syn(x)
        # print("x",x,"curr1",curr1,"self.readout_in_syn.weight",self.readout_in_syn.weight)
        self.spk,self.syn,self.mem=self.readout(curr1 , self.syn, self.mem)
        # print("self.mem",self.mem,"self.syn",self.syn,"self.spk",self.spk)
        return self.spk
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
network = net(N, 1, alpha=alpha, beta=beta).to(device)
# in_spike = torch.load('in_spike.pt')
in_spike = (torch.rand([T, N, 1]) > 0.5).float().to(device)
torch.save(in_spike, 'in_spike.pt')
stdp_learner = STDPLearner(synapse=network.readout_in_syn, sn=network.readout, tau_pre=tau_pre, tau_post=tau_post,
                                    f_pre=f_weight, f_post=f_weight).to(device)
out_spike = []
trace_pre = []
trace_post = []
weight = []
optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.)


with torch.no_grad():
    for t in range(T):
        optimizer.zero_grad()
        spk = network(in_spike[t])
        # print("in_spike[t]",in_spike[t])
        # print("spk:",spk)
        out_spike.append(spk.squeeze())
        stdp_learner(on_grad=True)  # 将STDP学习得到的参数更新量叠加到参数的梯度上
        # w = stdp_learner(on_grad=False)
        # print("w:",w)
        optimizer.step()
        weight.append(network.readout_in_syn.weight.data.clone().squeeze())
        trace_pre.append(stdp_learner.trace_pre.squeeze())
        trace_post.append(stdp_learner.trace_post.squeeze())


in_spike = in_spike.squeeze()
out_spike = torch.stack(out_spike)
trace_pre = torch.stack(trace_pre)
trace_post = torch.stack(trace_post)
weight = torch.stack(weight)
t = torch.arange(0, T).float()

in_spike = in_spike.to('cpu')
out_spike = out_spike.to('cpu')
trace_pre = trace_pre.to('cpu')
trace_post = trace_post.to('cpu')
weight = weight.to('cpu')
print(in_spike.shape)
cmap = plt.get_cmap('tab10')
plt.subplot(5, 1, 1)
plt.eventplot((in_spike * t)[in_spike == 1], lineoffsets=0, colors=cmap(0))
plt.xlim(-0.5, T + 0.5)
plt.ylabel('$s[i]$', rotation=0, labelpad=10)
plt.xticks([])
plt.yticks([])

plt.subplot(5, 1, 2)
plt.plot(t, trace_pre, c=cmap(1))
plt.xlim(-0.5, T + 0.5)
plt.ylabel('$tr_{pre}$', rotation=0)
plt.yticks([trace_pre.min().item(), trace_pre.max().item()])
plt.xticks([])

plt.subplot(5, 1, 3)
plt.eventplot((out_spike * t)[out_spike == 1], lineoffsets=0, colors=cmap(2))
plt.xlim(-0.5, T + 0.5)
plt.ylabel('$s[j]$', rotation=0, labelpad=10)
plt.xticks([])
plt.yticks([])

plt.subplot(5, 1, 4)
plt.plot(t, trace_post, c=cmap(3))
plt.ylabel('$tr_{post}$', rotation=0)
plt.yticks([trace_post.min().item(), trace_post.max().item()])
plt.xlim(-0.5, T + 0.5)
plt.xticks([])

plt.subplot(5, 1, 5)
plt.plot(t, weight, c=cmap(4))
plt.xlim(-0.5, T + 0.5)
plt.ylabel('$w[i][j]$', rotation=0)
plt.yticks([weight.min().item(), weight.max().item()])
plt.xlabel('time-step')

plt.gcf().subplots_adjust(left=0.18)

plt.show()