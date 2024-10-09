from matplotlib import pyplot as plt
from ReSuMe_models import *
import torch.nn as nn
import numpy as np

class net(nn.Module):
    def __init__(self,N,out_sz,alpha,beta):
        super().__init__()
        # input synapse and neuron
        # self.input_in_syn = nn.Linear(input_sz, N,bias=False)
        # self.input_in_syn.weight = nn.Parameter(torch.ones(N,input_sz))
        self.input = snn.Synaptic(alpha=alpha,beta=beta)
        self.syn, self.mem = self.input.init_synaptic()

        # ReSuMe learning rule
        self.ReSuMe = ReSuMe(N=N,out_sz=out_sz,alpha=alpha,beta=beta,train_flag=True)

    def forward(self,x):
        # curr1=self.input_in_syn(x)
        # print("curr1:",curr1)
        # print("self.syn:",self.input_in_syn)
        self.spk,self.syn,self.mem=self.input(x*0.05 , self.syn, self.mem)
        # print("x:",x,"\nself.spk:",self.spk)
        out = self.ReSuMe(self.spk)
        return out
    
if __name__ == "__main__":
    alpha = np.float32(np.exp(-1/16))
    beta = np.float32(1 - 1/16)
    sample_size = 5
    num_steps = 500
    batch_size = 10
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    network = net(2, 3, alpha=alpha, beta=beta).to(device)
    input_x = torch.tensor([[0,1],[0,0],[1,1],[0,0],[1,0]]).to(device).type(torch.float32)
    output_y = torch.tensor([[0,0,1],[0,0,0],[1,0,0],[0,0,0],[0,1,0]]).to(device).type(torch.float32)
    # input_x = torch.tensor([[1,1],[1,1],[0,1],[1,1],[1,1]]).to(device).type(torch.float32)
    # output_y = torch.tensor([[1,0,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0]]).to(device).type(torch.float32)
    # input_x = torch.tensor([[0,1],[1,0],[1,1],[0,1],[1,0]]).to(device).type(torch.float32)
    # output_y = torch.tensor([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[0,1,0]]).to(device).type(torch.float32)
    input_spk = []
    teach_spk = []
    teach_out_spk = []
    readout_spk = []
    weight = []
    teach_mem = []
    teach_syn = []
    mem = []
    syn = []
    x = []
    y = []
    out_collect = []
    with torch.no_grad():
        for j in range(input_x.shape[0]):
            x.append(input_x[j].repeat(batch_size,num_steps,1))
            y.append(output_y[j].repeat(batch_size,num_steps,1))
            for i in range(input_x.shape[1]):
                if input_x[j][i] == 1:
                    x[j][:,:,i] = (torch.rand_like(x[j][:,:,i]) > 0.0).float().to(device)
                else:
                    x[j][:,:,i] = (torch.rand_like(x[j][:,:,i]) > 1.0).float().to(device)

            for i in range(output_y.shape[1]):
                if output_y[j][i] == 1:
                    y[j][:,:,i] = (torch.rand_like(y[j][:,:,i]) > 0.0).float().to(device)
                else:
                    y[j][:,:,i] = (torch.rand_like(y[j][:,:,i]) > 1.0).float().to(device)
        for j in range(input_x.shape[0]):
            # readout init
            network.ReSuMe.readout.syn,network.ReSuMe.readout.mem = network.ReSuMe.readout.readout.init_synaptic()
            network.ReSuMe.readout.spk = None
            network.ReSuMe.readout.mask = 1.0
            # teach init
            network.ReSuMe.teach.syn,network.ReSuMe.teach.mem = network.ReSuMe.teach.teach.init_synaptic()
            network.ReSuMe.teach.spk = None

            for step in range(num_steps):
                # network.ReSuMe.readout.syn,network.ReSuMe.readout.mem = network.ReSuMe.readout.readout.init_synaptic()
                # network.ReSuMe.readout.spk = None
                # network.ReSuMe.readout.mask = 1.0
                network.ReSuMe.teach_input = y[j][:,step,:]
                print("teach_input:",network.ReSuMe.teach_input.shape)
                
                out = network(x[j][:,step,:])
                # print("x:",x[j][:step,:],"\nout",out)
                input_spk.append(x[j][:,step,:])
                teach_spk.append(y[j][:,step,:])
                readout_spk.append(out)
                teach_mem.append(network.ReSuMe.teach.mem)
                teach_syn.append(network.ReSuMe.teach.syn)
                mem.append(network.ReSuMe.readout.mem)
                syn.append(network.ReSuMe.readout.syn)
                teach_out_spk.append(network.ReSuMe.teach.spk)
                # print("readout_spk:",out)
                weight.append(network.ReSuMe.readout.readout_in_syn.weight.data.clone())
                print("weight:",network.ReSuMe.readout.readout_in_syn.weight.data.clone())
                # print(network.ReSuMe.readout.readout_in_syn.weight.data.clone())
                    # teach_spk = torch.stack(network.ReSuMe.teach.spk)
                    # readout_spk = torch.stack(network.ReSuMe.readout.spk)
                    # weight = network.ReSuMe.readout.readout_in_syn.weight.data.clone().squeeze()
        # print("teach_spk:",teach_spk)
    weight = torch.stack(weight)
    teach_spk = torch.stack(teach_spk)
    readout_spk = torch.stack(readout_spk)
    input_spk = torch.stack(input_spk)
    mem = torch.stack(mem)
    syn = torch.stack(syn)
    teach_mem = torch.stack(teach_mem)
    teach_syn = torch.stack(teach_syn)
    teach_out_spk = torch.stack(teach_out_spk)
    # print("weight:",weight.shape)
    # print("teach_spk:",teach_spk.shape)
    # print("readout_spk:",readout_spk.shape)
    # print("input_spk:",input_spk.shape)
    weight = weight.to('cpu')
    teach_spk = teach_spk.to('cpu')
    readout_spk = readout_spk.to('cpu')
    input_spk = input_spk.to('cpu')
    mem = mem.to('cpu')
    syn = syn.to('cpu')
    teach_mem = teach_mem.to('cpu')
    teach_syn = teach_syn.to('cpu')
    teach_out_spk = teach_out_spk.to('cpu')
    teach_spk = teach_spk.reshape(teach_spk.size(0)*teach_spk.size(1),-1)
    readout_spk = readout_spk.reshape(readout_spk.size(0)*readout_spk.size(1),-1)
    weight = weight.reshape(weight.size(0),-1)
    mem = mem.reshape(mem.size(0)*mem.size(1),-1)
    input_spk = input_spk.reshape(input_spk.size(0)*input_spk.size(1),-1)
    syn = syn.reshape(syn.size(0)*syn.size(1),-1)
    teach_mem = teach_mem.reshape(teach_mem.size(0)*teach_mem.size(1),-1)
    teach_syn = teach_syn.reshape(teach_syn.size(0)*teach_syn.size(1),-1)
    teach_out_spk = teach_out_spk.reshape(teach_out_spk.size(0)*teach_out_spk.size(1),-1)
    # print("weight:",weight.shape)
    t = torch.arange(0,batch_size*num_steps*sample_size).float()
    t_weight = torch.arange(0,num_steps*sample_size).float()
    
    readout_spk_test = []
    mem_test = []
    syn_test = []
    for j in range(input_x.shape[0]):
                x.append(input_x[j].repeat(1,num_steps,1))
                y.append(output_y[j].repeat(1,num_steps,1))
    network.ReSuMe.train_flag = False
    print("mask:",network.ReSuMe.readout.mask)
    # network.ReSuMe.readout.readout_in_syn.weight.data = torch.tensor([[0.5,0.5],[1.0,-1.0],[-1.0,1.0]]).to(device).type(torch.float32)
    with torch.no_grad():
        for j in range(input_x.shape[0]):
            out_collect = []
            # readout init
            network.ReSuMe.readout.syn,network.ReSuMe.readout.mem = network.ReSuMe.readout.readout.init_synaptic()
            network.ReSuMe.readout.spk = None
            network.ReSuMe.readout.mask = 1.0
            # teach init
            network.ReSuMe.teach.syn,network.ReSuMe.teach.mem = network.ReSuMe.teach.teach.init_synaptic()
            network.ReSuMe.teach.spk = None
            for step in range(num_steps):
                # network.ReSuMe.readout.syn,network.ReSuMe.readout.mem = network.ReSuMe.readout.readout.init_synaptic()
                # network.ReSuMe.readout.spk = None
                # network.ReSuMe.readout.mask = 1.0
                out = network(x[j][:,step,:])
                readout_spk_test.append(out)
                mem_test.append(network.ReSuMe.readout.mem)
                syn_test.append(network.ReSuMe.readout.syn)
                out_collect.append(out)
            
            out_collect = torch.stack(out_collect)
            out_collect = out_collect.to('cpu')
            out_collect = out_collect.reshape(out_collect.size(0)*out_collect.size(1),-1)
            print("out_collect mean:",torch.mean(out_collect,dim=0))
            print("x:",torch.mean(x[j][:,step,:],dim=0),"\nout",torch.mean(out,dim=0))
            print("y:",torch.mean(y[j][:,step,:],dim=0))
            # print("x:",x[j][5,step,:],"\nout",out[5])
            # print("y:",y[j][5,step,:])
        print("weight:",network.ReSuMe.readout.readout_in_syn.weight.data.clone())
    mem_test = torch.stack(mem_test)
    syn_test = torch.stack(syn_test)
    readout_spk_test = torch.stack(readout_spk_test)
    mem_test = mem_test.to('cpu')
    syn_test = syn_test.to('cpu')
    readout_spk_test = readout_spk_test.to('cpu')
    mem_test = mem_test.reshape(mem_test.size(0)*mem_test.size(1),-1)
    syn_test = syn_test.reshape(syn_test.size(0)*syn_test.size(1),-1)
    readout_spk_test = readout_spk_test.reshape(readout_spk_test.size(0)*readout_spk_test.size(1),-1)


            # print("diff:",(y[j][:,step,:]-out)[0])


    cmap = plt.get_cmap('tab10')
    figure = plt.figure(0)
    for i in range(3):
        spk = teach_spk[:,i]
        plt.subplot(3, 1, i+1)
        plt.eventplot((spk*t)[spk==1], lineoffsets=0, colors=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$teach$', rotation=0, labelpad=10)
        plt.xticks([])
        plt.yticks([])

    figure = plt.figure(1)
    for i in range(3):
        spk = readout_spk[:,i]
        plt.subplot(3, 1, i+1)
        plt.eventplot((spk*t)[spk==1], lineoffsets=0, colors=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$readout$', rotation=0, labelpad=10)
        plt.xticks([])
        plt.yticks([])

    figure = plt.figure(2)
    for i in range(6):
        w = weight[:,i]
        plt.subplot(6, 1, i+1)
        plt.plot(t_weight, w, c=cmap(4))
        plt.xlim(-0.5, sample_size*num_steps + 0.5)
        plt.ylabel('$weight$', rotation=0)
        plt.yticks([w.min().item(), w.max().item()])
        plt.xlabel('time-step')

    figure = plt.figure(3)
    for i in range(2):
        spk = input_spk[:,i]
        plt.subplot(2, 1, i+1)
        plt.eventplot((spk*t)[spk==1], lineoffsets=0, colors=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$input$', rotation=0, labelpad=10)
        plt.xticks([])
        plt.yticks([])

    figure = plt.figure(4)
    for i in range(3):
        m = mem[:,i]
        plt.subplot(3, 1, i+1)
        plt.plot(t,m,c=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$mem$', rotation=0, labelpad=10)
        plt.xticks([])
        plt.yticks([])

    figure = plt.figure(5)
    for i in range(3):
        s = syn[:,i]
        plt.subplot(3, 1, i+1)
        plt.plot(t,s,c=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$syn$', rotation=0)
    
    figure = plt.figure(6)
    for i in range(3):
        m_t = mem_test[:,i]
        plt.subplot(3, 1, i+1)
        plt.plot(t,m_t,c=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$mem_test$', rotation=0)
    
    figure = plt.figure(7)
    for i in range(3):
        s_t = syn_test[:,i]
        plt.subplot(3, 1, i+1)
        plt.plot(t,s_t,c=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$syn_test$', rotation=0)

    figure = plt.figure(8)
    for i in range(3):
        spk = readout_spk_test[:,i]
        plt.subplot(3, 1, i+1)
        plt.eventplot((spk*t)[spk==1], lineoffsets=0, colors=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$readout_test$', rotation=0, labelpad=10)
        plt.xticks([])
        plt.yticks([])
    figure = plt.figure(9)
    for i in range(3):
        t_m = teach_mem[:,i]
        plt.subplot(3, 1, i+1)
        plt.plot(t,t_m,c=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$teach_mem$', rotation=0)
    figure = plt.figure(10)
    for i in range(3):
        t_s = teach_syn[:,i]
        plt.subplot(3, 1, i+1)
        plt.plot(t,t_s,c=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$teach_syn$', rotation=0)
    figure = plt.figure(11)
    for i in range(3):
        spk = teach_out_spk[:,i]
        plt.subplot(3, 1, i+1)
        plt.eventplot((spk*t)[spk==1], lineoffsets=0, colors=cmap(i))
        plt.xlim(-0.5, batch_size*num_steps*sample_size + 0.5)
        plt.ylabel('$teach_out$', rotation=0, labelpad=10)

    plt.show()