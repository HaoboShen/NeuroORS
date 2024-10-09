import csv
import serial
import threading
import queue
import torch
import time
import numpy as np
from lsm_models import LSM

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

class serial_class():
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.receive_thread_running = False
        self.file_path = "%s.csv"%(time.strftime("test_data_%Y-%m-%d-%H-%M-%S", time.localtime()))
        self.serial = None
        self.write_to_file_flag = True
        self.data_queue = queue.Queue()
        self.data_raw = ""
        self.data_str = []
    
    def receive_start(self):
        self.serial = serial.Serial(self.port, self.baudrate)
        self.receive_thread_ = threading.Thread(target=self.receive_thread)
        self.receive_thread_running = True
        self.receive_thread_.start()
    
    def receive_stop(self):
        self.receive_thread_running = False
        # self.receive_thread_.join()
        self.serial.close()

    def receive_thread(self):
        while self.receive_thread_running:
            try:
                if self.serial.in_waiting > 0 and self.serial.is_open:
                    data = self.serial.read(self.serial.in_waiting).decode()
                    self.decode_data(data)
            except serial.SerialException as e:
                print(e)
                self.receive_thread_running = False
                self.serial.close()
                break

    def decode_data(self,data):
        # 解析数据
        self.data_raw += data
        if len(self.data_raw) > 200:
            self.data_raw = self.data_raw.replace("\r\n","")
            self.data_raw = self.data_raw.replace("\n","")
            # print("data_raw:",len(self.data_raw),"data_raw:",self.data_raw)
            self.data_str += self.data_raw.split(" ")
            self.data_str = [str for str in self.data_str if str != ""]
            # print("data_str:",self.data_str,"data_str_len:",len(self.data_str))
            self.data_raw = ""
        if len(self.data_str) >= 40 :
            if self.data_str[0] == "1" and self.data_str[1] == "1":
                data_float = list(map(float,self.data_str[2:36]))
                # if self.write_to_file_flag:
                #     self.write_to_file_signal.emit(self.data_str[:40])
                data_to_predict = data_float[2:26]
                # print(data_draw)
                
                self.data_queue.put(data_to_predict)
                self.data_str = self.data_str[41:]
            else:
                self.data_str.pop(0)

if __name__ == '__main__':
    ser = serial_class('COM5', 115200)
    ser.receive_start()
    num_step = 5
    scale = 2500
    data_tensor = torch.empty((0,num_step,24))
    data_tensor_line = torch.empty((0,24))
    data_tensor_diff = torch.zeros((1,num_step,24))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    alpha = np.float32(np.exp(-1/16))
    beta = np.float32(1 - 1/16)

    load_model_name = "lsm_net_orange_peach_online_diff_A.pth"
    teach_flag = False
    teach_target = torch.tensor([0]).to(device)
    update_weight_flag = False

    lsm_net = LSM(1000, 24, 3, np.float32(np.zeros((1000,24))), np.float32(np.zeros((1000,1000))), alpha=alpha, beta=beta, th=20).to(device)
    lsm_net.load_state_dict(torch.load(load_model_name))
    lsm_net.eval()
    print(lsm_net)

    ten_results = []
    count_result = 0

# print("lsm_out:", lsm_out)
    while ser.receive_thread_running:
        if ser.data_queue.qsize() > 0:
            # print(ser.data_queue.qsize())
            data = ser.data_queue.get()
            data_tensor_line = torch.tensor(data).repeat(1,1,1)
            # print(data_tensor_line.shape)
            if data_tensor.shape[0] < 1:
                data_tensor = data_tensor_line.repeat(1,num_step,1)
            else:
                data_tensor = data_tensor[:,1:,:]
                data_tensor_diff = data_tensor_diff[:,1:,:]
                data_tensor_diff_line = data_tensor_line-data_tensor[:,-1,:]

                data_tensor = torch.cat((data_tensor,data_tensor_line),dim=1)
                data_tensor_diff = torch.cat((data_tensor_diff,data_tensor_diff_line),dim=1)
                # print("data_tensor_diff:",data_tensor_diff)
                
                # print("data_tensor:",data_tensor)
                # print("data_tensor_line:",data_tensor_line)
            # print(data_tensor)
            # print(data_tensor.shape)
            # data_tensor_device = data_tensor.to(device).type(torch.float32)*scale
            # print(data_tensor_device)
            # print(data_tensor_device.shape)
            data_tensor_device = data_tensor_diff.to(device).type(torch.float32)*scale
            
            print(data_tensor_device)
            print(data)
            print(data_tensor_device.shape)
            
            start_time = time.time()
            spk_rec = lsm_net(data_tensor_device)
            # print("spk_rec",spk_rec)
            end_time = time.time()
            spk_mean = torch.mean(spk_rec,dim=0)
            # print("spk:",spk_rec)
            print("spk_mean:",spk_mean)
            spk_max,spk_max_idx = torch.max(spk_mean,1)
            print("spk_max_idx:",spk_max_idx)            # update weights
            if spk_max_idx[0] != 0 and update_weight_flag:
                targets = targets_to_spike(spk_max_idx,num_steps=num_step,out_sz=3).to(device)
                lsm_net.teach_input = targets
                lsm_net.ReSuMe.train_flag = True
                spk_rec = lsm_net(data_tensor_device)
                lsm_net.ReSuMe.train_flag = False
                print("update weights")
                torch.save(lsm_net.state_dict(), load_model_name)

            # teach
            if teach_flag:
                
                targets = targets_to_spike(teach_target,num_steps=num_step,out_sz=3).to(device)
                lsm_net.teach_input = targets
                lsm_net.ReSuMe.train_flag = True
                spk_rec = lsm_net(data_tensor_device)
                lsm_net.ReSuMe.train_flag = False
                print("weight:",lsm_net.ReSuMe.readout.readout_in_syn.weight)
                print("update weights")
                torch.save(lsm_net.state_dict(), load_model_name)

            if ser.write_to_file_flag:
                with open(ser.file_path,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data+spk_max_idx.tolist())
            # results analze
            ten_results.append(spk_max_idx[0].item())
            if len(ten_results)>=11:
                ten_results = ten_results[1:]
            count_result = max(set(ten_results), key=ten_results.count)
            print("ten_result:",ten_results)
            print("count_result:",count_result)

            
            if spk_max_idx[0] == 0.0:
                print("air")
            elif spk_max_idx[0] == 1.0:
                print("orange")
            elif spk_max_idx[0] == 2.0:
                print("peach")
            else:
                print("unknown")
