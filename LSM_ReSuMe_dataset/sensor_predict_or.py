import csv
import serial
import threading
import queue
import torch
import time
import numpy as np
from lsm_models import LSM

class serial_class():
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.receive_thread_running = False
        self.serial = None
        self.write_to_file_flag = True
        self.file_path = "%s.csv"%(time.strftime("test_data_%Y-%m-%d-%H-%M-%S", time.localtime()))
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
    ser = serial_class('COM4', 115200)
    ser.receive_start()
    num_step = 10
    scale = 1.0
    data_tensor = torch.empty((0,num_step,24))
    data_tensor_line = torch.empty((0,24))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    alpha = np.float32(np.exp(-1/16))
    beta = np.float32(1 - 1/16)

    lsm_net = LSM(1000, 24, 3, np.float32(np.zeros((1000,24))), np.float32(np.zeros((1000,1000))), alpha=alpha, beta=beta, th=20).to(device)
    lsm_net.load_state_dict(torch.load("lsm_net_orange_peach.pth"))
    lsm_net.eval()
    print(lsm_net)

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

                data_tensor = torch.cat((data_tensor,data_tensor_line),dim=1)
                print("data_tensor:",data_tensor)
                print("data_tensor_line:",data_tensor_line)
            # print(data_tensor)
            # print(data_tensor.shape)
            # data_tensor_device = data_tensor.to(device).type(torch.float32)*scale
            # print(data_tensor_device)
            # print(data_tensor_device.shape)
            data_tensor_device = data_tensor.to(device).type(torch.float32)*scale
            print(data_tensor_device)
            print(data_tensor_device.shape)
            

        
        # if ser.data_queue.qsize() >= num_step:
        #     print("qsize:",ser.data_queue.qsize())
        #     # print("data1:",ser.data_queue.get())
        #     # print("data2:",ser.data_queue.get())
        #     # print("data3:",ser.data_queue.get())
        #     for i in range(num_step):
        #         data_tensor = torch.concat((data_tensor,torch.tensor(ser.data_queue.get()).unsqueeze(0)),dim=0)
        #     # print("after qsize:",ser.data_queue.qsize())
        #     data_tensor = data_tensor.unsqueeze(0)
        #     print(data_tensor)
        #     data_tensor_device = data_tensor.to(device).type(torch.float32)
        #     data_tensor = torch.empty((0,24))
        #     print(data_tensor.shape)
            start_time = time.time()
            spk_rec = lsm_net(data_tensor_device)
            # print("spk_rec",spk_rec)
            end_time = time.time()
            spk_mean = torch.mean(spk_rec,dim=0)
            spk_max,spk_max_idx = torch.max(spk_mean,1)
            # print("spk_rec",spk_rec)
            print("spk:",spk_mean)
            if ser.write_to_file_flag:
                with open(ser.file_path,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data+spk_max_idx.tolist())
            print("spk_max_idx:",spk_max_idx)
            # if spk_max_idx[0] == 0.0*scale:
            #     print("air")
            # elif spk_max_idx[0] == 1.0*scale:
            #     print("banana")
            # elif spk_max_idx[0] == 2.0*scale:
            #     print("orange")
            # elif spk_max_idx[0] == 3.0*scale:
            #     print("peach")
            # else:
            #     print("unknown")

            # if spk_max_idx[0] == 0.0:
            #     print("air")
            # elif spk_max_idx[0] == 1.0:
            #     print("alcohol")
            # elif spk_max_idx[0] == 2.0:
            #     print("vinegar")
            # else:
            #     print("unknown")

            # if spk_max_idx[0] == 0.0*scale:
            #     print("air")
            # elif spk_max_idx[0] == 1.0*scale:
            #     print("alcohol")
            # elif spk_max_idx[0] == 2.0*scale:
            #     print("apple")
            # elif spk_max_idx[0] == 3.0*scale:
            #     print("banana")
            # elif spk_max_idx[0] == 4.0*scale:
            #     print("blueberry")
            # elif spk_max_idx[0] == 5.0*scale:
            #     print("orange")
            # elif spk_max_idx[0] == 6.0*scale:
            #     print("peach")
            # elif spk_max_idx[0] == 7.0*scale:
            #     print("vinegar")
            # elif spk_max_idx[0] == 8.0*scale:
            #     print("watermelon")
            # else:
            #     print("unknown")
            # if spk_max_idx[0] == 0.0*scale:
            #     print("air")
            # elif spk_max_idx[0] == 1.0*scale:
            #     print("orange")
            # elif spk_max_idx[0] == 2.0*scale:
            #     print("watermelon")
            # else:
            #     print("unknown")
            if spk_max_idx[0] == 0.0*scale:
                print("air")
            elif spk_max_idx[0] == 1.0*scale:
                print("orange")
            elif spk_max_idx[0] == 2.0*scale:
                print("peach")
            else:
                print("unknown")