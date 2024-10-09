import csv
import os
import serial
import threading
import queue
import torch
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from lsm_weight_definitions import *
from ultralytics import YOLO
from lsm_models import LSM
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

class yolo_fruits_detect():
    def __init__(self):
        self.view_img = True
        # Load model
        self.model = YOLO('yolov8m_fruits.pt')  # load modelpyt
        print("model:",self.model)

        self.pipeline = rs.pipeline()
        # 创建 config 对象：
        self.config = rs.config()
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

        # Start streaming
        self.pipeline.start(self.config)
        self.align_to_color = rs.align(rs.stream.color)
        self.detect_running = True
        self.detect_thread_ = threading.Thread(target=self.detect)
        self.detect_thread_.start()

        self.output = None
        self.color_image = None

    def detect(self):
        while self.detect_running:
            # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
            frames = self.pipeline.wait_for_frames()
            frames = self.align_to_color.process(frames)
            # depth_frame = frames.get_depth_frame()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())
            self.output = self.model(self.color_image,verbose=False)
        self.pipeline.stop()
        cv2.destroyAllWindows()

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
    ser = serial_class('COM3', 115200)
    ser.receive_start()
    num_step = 10
    scale = 2500
    data_tensor = torch.empty((0,num_step,24))
    data_tensor_line = torch.empty((0,24))
    data_tensor_diff = torch.zeros((1,num_step,24))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    curr_prefac = np.float32(1/16)
    alpha = np.float32(np.exp(-1/16))
    beta = np.float32(1 - 1/16)

    yolo = yolo_fruits_detect()
    to_pil_image = transforms.ToPILImage()

    online_learning_flag = False

    load_model_name = "lsm_net_diff_yolo.pth"
    teach_flag = True

    teach_target = torch.tensor([2]).to(device)
    # update_weight_flag = False
    Win, Wlsm = initWeights1(LqWin=27, LqWlsm=2, in_conn_density=0.15, in_size=24, lam=9, 
                            inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None)
    
    lsm_net = LSM(1000, 24, 3, np.float32(np.zeros((1000,24))), np.float32(np.zeros((1000,1000))), alpha=alpha, beta=beta, th=20).to(device)
    if os.path.exists(load_model_name):
        lsm_net.load_state_dict(torch.load(load_model_name))
        print("load model")
    else:
        lsm_net = LSM(1000, 24, 3, np.float32(curr_prefac*Win), np.float32(curr_prefac*Wlsm), alpha=alpha, beta=beta, th=20).to(device)
        print("create model")
    lsm_net.eval()
    print(lsm_net)

    ten_results = []
    count_result = 0

    odor_result = ["",(255,255,255)]

    while ser.receive_thread_running:
        # image
        if yolo.output is not None:
            im0 = yolo.output[0].plot()
        elif yolo.color_image is not None:
            im0 = yolo.color_image
        else:
            im0 = np.zeros((480,640,3))

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

            data_tensor_device = data_tensor_diff.to(device).type(torch.float32)*scale
            
            print(data_tensor_device)
            print(data)
            print(data_tensor_device.shape)

            start_time = time.time()            # show as image
            data_image = data_tensor_device[0,:,:]
            pil_image = to_pil_image(data_image)
            # cv_image = cv2.cvtColor(np.array(pil_image))
            heatmap = cv2.applyColorMap(np.array(pil_image), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap,(24*20,num_step*20),interpolation=cv2.INTER_NEAREST)

            spk_rec = lsm_net(data_tensor_device)

            # print("spk_rec",spk_rec)
            end_time = time.time()
            spk_mean = torch.mean(spk_rec,dim=0)
            # print("spk:",spk_rec)
            print("spk_mean:",spk_mean)
            spk_max,spk_max_idx = torch.max(spk_mean,1)
            print("spk_max_idx:",spk_max_idx)            # update weights
            cv2.putText(heatmap, "%d"%spk_max_idx[0], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("data_image",heatmap)

            if spk_max_idx[0] == 0.0:
                print("air")
            elif spk_max_idx[0] == 1.0:
                print("orange")
            elif spk_max_idx[0] == 2.0:
                print("peach")
            else:
                print("unknown")
            # get yolo result as teach target
            
            if yolo.output is not None and online_learning_flag:
                cls = yolo.output[0].boxes.cls
                if len(cls) == 1:
                    if cls[0] == 13:
                        if spk_max_idx[0] != 1:
                            teach_target = torch.tensor([1]).to(device)
                            teach_flag = True
                    elif cls[0] == 14:
                        if spk_max_idx[0] != 2 :
                            teach_target = torch.tensor([2]).to(device)
                            teach_flag = True
                elif len(cls) == 0 and spk_max_idx[0] == 0:
                    teach_target = torch.tensor([0]).to(device)
                    teach_flag = True
                else:
                    teach_flag = False
                print("teach_target:",teach_target,"teach_flag:",teach_flag)
            # record weight
            weight_pre = lsm_net.ReSuMe.readout.readout_in_syn.weight
            # teach
            if teach_flag:
                targets = targets_to_spike(teach_target,num_steps=num_step,out_sz=3).to(device)
                
                # print("targets:",targets)
                lsm_net.teach_input = targets
                lsm_net.ReSuMe.train_flag = True
                spk_rec = lsm_net(data_tensor_device)
                lsm_net.ReSuMe.train_flag = False
                
                print("weight:",lsm_net.ReSuMe.readout.readout_in_syn.weight)
                print("update weights")
                torch.save(lsm_net.state_dict(), load_model_name)
                teach_flag = False

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

            if count_result == 0.0:
                odor_result = ["air",(255,255,255)]
            elif count_result == 1.0:
                odor_result = ["orange",(0,165,255)]
            elif count_result == 2.0:
                odor_result = ["peach",(203,192,255)]
            else:
                print("unknown")
        # weight image
        weight_diff = lsm_net.ReSuMe.readout.readout_in_syn.weight - weight_pre
        weight_img0 = to_pil_image(weight_diff[0,:].reshape(25,40))
        weight_img1 = to_pil_image(weight_diff[1,:].reshape(25,40))
        weight_img2 = to_pil_image(weight_diff[2,:].reshape(25,40))
        # cv_image = cv2.cvtColor(np.array(pil_image))
        heatmap0 = cv2.applyColorMap(np.array(weight_img0), cv2.COLORMAP_JET)
        heatmap0 = cv2.resize(heatmap0,(40*10,25*10),interpolation=cv2.INTER_NEAREST)
        cv2.imshow("weight_diff0",heatmap0)
        heatmap1 = cv2.applyColorMap(np.array(weight_img1), cv2.COLORMAP_JET)
        heatmap1 = cv2.resize(heatmap1,(40*10,25*10),interpolation=cv2.INTER_NEAREST)
        cv2.imshow("weight_diff1",heatmap1)
        heatmap2 = cv2.applyColorMap(np.array(weight_img2), cv2.COLORMAP_JET)
        heatmap2 = cv2.resize(heatmap2,(40*10,25*10),interpolation=cv2.INTER_NEAREST)
        cv2.imshow("weight_diff2",heatmap2)


        # view image
        if yolo.view_img:
            cv2.putText(im0, odor_result[0], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,odor_result[1], 2)
            cv2.imshow("detection", im0)
            
            if cv2.waitKey(1) == ord('q'):
                yolo.detect_running = False  # q to quit
                break
