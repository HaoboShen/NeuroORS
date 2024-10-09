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
import torch.nn as nn

def targets_to_spike(ts, num_steps, out_sz, active_rate=1.0, inactive_rate=0.0):
    spike_train = torch.zeros(ts.shape[0], num_steps, out_sz)
    for i in range(out_sz):
        spike_train[:,:,i] = (torch.rand_like(spike_train[:,:,i]) > 1.0-inactive_rate).float()
    for j in range(ts.shape[0]):
        if ts[j] == 0.0:
            spike_train[j,:,0] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 1.0:
            spike_train[j,:,1] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 2.0:
            spike_train[j,:,2] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
        if ts[j] == 3.0:
            spike_train[j,:,3] = (torch.rand_like(spike_train[j,:,0]) > 1.0-active_rate).float()
    return spike_train

class yolo_fruits_detect():
    def __init__(self):
        self.view_img = True
        self.model = YOLO('yolov8m_fruits.pt')  # load model
        print("model:", self.model)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

        self.pipeline.start(self.config)
        self.align_to_color = rs.align(rs.stream.color)
        self.detect_running = True
        self.detect_thread_ = threading.Thread(target=self.detect)
        self.detect_thread_.start()

        self.output = None
        self.color_image = None

    def detect(self):
        while self.detect_running:
            frames = self.pipeline.wait_for_frames()
            frames = self.align_to_color.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())
            self.output = self.model(self.color_image, verbose=False)
        self.pipeline.stop()
        cv2.destroyAllWindows()

class serial_class():
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.receive_thread_running = False
        self.file_path = "%s.csv" % (time.strftime("test_data_%Y-%m-%d-%H-%M-%S", time.localtime()))
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

    def decode_data(self, data):
        self.data_raw += data
        if len(self.data_raw) > 200:
            self.data_raw = self.data_raw.replace("\r\n", "")
            self.data_raw = self.data_raw.replace("\n", "")
            self.data_str += self.data_raw.split(" ")
            self.data_str = [str for str in self.data_str if str != ""]
            self.data_raw = ""
        if len(self.data_str) >= 40:
            if self.data_str[0] == "1" and self.data_str[1] == "1":
                data_float = list(map(float, self.data_str[2:36]))
                data_to_predict = data_float[2:26]
                self.data_queue.put(data_to_predict)
                self.data_str = self.data_str[41:]
            else:
                self.data_str.pop(0)

if __name__ == '__main__':
    ser = serial_class('COM3', 115200)
    ser.receive_start()
    num_step = 10
    scale = 2500
    data_tensor = torch.empty((0, num_step, 24))
    data_tensor_line = torch.empty((0, 24))
    data_tensor_diff = torch.zeros((1, num_step, 24))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    curr_prefac = np.float32(1 / 16)
    alpha = np.float32(np.exp(-1 / 16))
    beta = np.float32(1 - 1 / 16)

    yolo = yolo_fruits_detect()
    to_pil_image = transforms.ToPILImage()

    online_learning_flag = True

    load_model_name = "lsm_net_diff_yolo.pth"
    teach_flag = True

    teach_target = torch.tensor([2]).to(device)
    Win, Wlsm = initWeights1(LqWin=27, LqWlsm=2, in_conn_density=0.15, in_size=24, lam=9,
                            inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None)
    
    lsm_net = LSM(1000, 24, 3, np.float32(np.zeros((1000, 24))), np.float32(np.zeros((1000, 1000))), alpha=alpha, beta=beta, th=20).to(device)
    if os.path.exists(load_model_name):
        lsm_net.load_state_dict(torch.load(load_model_name))
        print("load model")
    else:
        lsm_net = LSM(1000, 24, 3, np.float32(curr_prefac * Win), np.float32(curr_prefac * Wlsm), alpha=alpha, beta=beta, th=20).to(device)
        print("create model")
        lsm_net.ReSuMe.readout.readout_in_syn.weight = nn.Parameter(torch.ones(3, 1000).to(device) * -1.0)
    lsm_net.eval()
    print(lsm_net)

    ten_results = []
    odor_result = ["", (255, 255, 255)]

    timestamps = []
    weights_over_time = [[] for _ in range(3)]  # Separate weight storage for each neuron

    start_time = time.time()
    while ser.receive_thread_running:

        if yolo.output is not None:
            im0 = yolo.output[0].plot()
        elif yolo.color_image is not None:
            im0 = yolo.color_image
        else:
            im0 = np.zeros((480, 640, 3))

        if ser.data_queue.qsize() > 0:
            data = ser.data_queue.get()
            data_tensor_line = torch.tensor(data).repeat(1, 1, 1)
            if data_tensor.shape[0] < 1:
                data_tensor = data_tensor_line.repeat(1, num_step, 1)
            else:
                data_tensor = data_tensor[:, 1:, :]
                data_tensor_diff = data_tensor_diff[:, 1:, :]
                data_tensor_diff_line = data_tensor_line - data_tensor[:, -1, :]

                data_tensor = torch.cat((data_tensor, data_tensor_line), dim=1)
                data_tensor_diff = torch.cat((data_tensor_diff, data_tensor_diff_line), dim=1)
            print("data_tensor:", data_tensor)
            data_tensor_device = data_tensor_diff.to(device).type(torch.float32) * scale
            
            spk_rec = lsm_net(data_tensor_device)
            spk_mean = torch.mean(spk_rec, dim=0)
            spk_max, spk_max_idx = torch.max(spk_mean, 1)

            data_image = data_tensor_device[0, :, :]
            pil_image = to_pil_image(data_image)
            heatmap = cv2.applyColorMap(np.array(pil_image), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (24 * 20, num_step * 20), interpolation=cv2.INTER_NEAREST)
            cv2.putText(heatmap, "%d" % spk_max_idx[0], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("data_image", heatmap)
            print("spk_mean:", spk_mean)

            if spk_max_idx[0] == 0.0:
                print("air")
            elif spk_max_idx[0] == 1.0:
                print("orange")
            elif spk_max_idx[0] == 2.0:
                print("peach")
            else:
                print("unknown")

            if yolo.output is not None and online_learning_flag:
                cls = yolo.output[0].boxes.cls
                if len(cls) == 1:
                    if cls[0] == 13:
                        if spk_max_idx[0] != 1:
                            teach_target = torch.tensor([1]).to(device)
                            teach_flag = True
                        # teach_target = torch.tensor([1]).to(device)
                        # teach_flag = True
                    elif cls[0] == 14:
                        if spk_max_idx[0] != 2:
                            teach_target = torch.tensor([2]).to(device)
                            teach_flag = True
                        # teach_target = torch.tensor([2]).to(device)
                        # teach_flag = True
                elif len(cls) == 0 and spk_max_idx[0] == 0:
                # elif len(cls) == 0 :
                    teach_target = torch.tensor([0]).to(device)
                    teach_flag = True
                else:
                    teach_flag = False
                print("teach_target:", teach_target, "teach_flag:", teach_flag)            
            
            weight_abs = lsm_net.ReSuMe.readout.readout_in_syn.weight.clone().detach().cpu().numpy()
            print("weight_abs:", weight_abs)
            
            timestamps.append(time.time())
            for i in range(3):
                weights_over_time[i].append(weight_abs[i])

            if teach_flag:
                targets = targets_to_spike(teach_target, num_steps=num_step, out_sz=3).to(device)
                lsm_net.teach_input = targets
                lsm_net.ReSuMe.train_flag = True
                spk_rec = lsm_net(data_tensor_device)
                lsm_net.ReSuMe.train_flag = False
                print("update weights")
                torch.save(lsm_net.state_dict(), load_model_name)
                teach_flag = False

            if ser.write_to_file_flag:
                with open(ser.file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data + spk_max_idx.tolist())
            
            ten_results.append(spk_max_idx[0].item())
            if len(ten_results) >= 11:
                ten_results = ten_results[1:]
            count_result = max(set(ten_results), key=ten_results.count)
            print("ten_result:", ten_results)
            print("count_result:", count_result)

            if count_result == 0.0:
                odor_result = ["air", (255, 255, 255)]
            elif count_result == 1.0:
                odor_result = ["orange", (0, 165, 255)]
            elif count_result == 2.0:
                odor_result = ["peach", (203, 192, 255)]
            else:
                print("unknown")

        if yolo.view_img:
            cv2.putText(im0, odor_result[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, odor_result[1], 2)
            cv2.imshow("detection", im0)

            if cv2.waitKey(1) == ord('q'):
                yolo.detect_running = False  # q to quit
                ser.receive_thread_running = False
                cv2.destroyAllWindows()
                break
    end_time = time.time()
    ser.receive_stop()
    print("time:", end_time - start_time)

    # Save weights changes to a single CSV file
    with open("weights_over_time.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = ["timestamp"] + [f"neuron_{i}_synapse_{j}" for i in range(3) for j in range(25 * 40)]
        csvwriter.writerow(headers)
        for timestamp, weights0, weights1, weights2 in zip(timestamps, weights_over_time[0], weights_over_time[1], weights_over_time[2]):
            row = [timestamp] + weights0.tolist() + weights1.tolist() + weights2.tolist()
            csvwriter.writerow(row)

    print("Weights over time have been saved to 'weights_over_time.csv'.")
