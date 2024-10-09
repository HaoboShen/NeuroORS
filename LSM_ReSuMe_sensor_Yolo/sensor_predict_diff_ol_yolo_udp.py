import queue
import socket
import threading
import time
import cv2
import torchvision.transforms as transforms
import torch
import numpy as np
import pyrealsense2 as rs
from lsm_weight_definitions import *
import serial
from lsm_models import LSM
from ultralytics import YOLO


    
def udp_send(data):
    global data_to_send
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('192.168.85.130', 10000)
    print("data:",data)
    server_socket.sendto(np.array(data).tobytes(), server_address)
    #time.sleep(1)
    server_socket.close()

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
        global count_resultr
        data_pre = []
        while self.detect_running:
            # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
            frames = self.pipeline.wait_for_frames()
            frames = self.align_to_color.process(frames)
            # depth_frame = frames.get_depth_frame()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())
            self.output = self.model(self.color_image,verbose=False)
            boxes = self.output[0].boxes
            for i,cls in enumerate(boxes.cls):
                # if cls == 14: #13: orange 14: peach 5: apple
                # center = [(boxes.xyxy[i][0]+boxes.xyxy[i][2]).item()/2,(boxes.xyxy[i][1]+boxes.xyxy[i][3]).item()/2]
                center = [(boxes.xyxy[i][0]+boxes.xyxy[i][2]).item()/2,(boxes.xyxy[i][1]+boxes.xyxy[i][3]).item()/2]
                if count_result == 1.0 and cls.item() == 13.0 and data_pre[0] != 1.0:
                    dis = depth_frame.get_distance(int(center[0]),int(center[1]) )  # 获取该像素点对应的深度
                    data = [1.0]+rs.rs2_deproject_pixel_to_point(depth_intrin, center, dis)
                    #data[1] = data[1] - 0.0425 #对齐r
                    udp_send(data)
                elif count_result == 2.0 and cls.item() == 14.0 and data_pre[0] != 2.0:
                    dis = depth_frame.get_distance(int(center[0]),int(center[1]) )
                    data = [2.0]+rs.rs2_deproject_pixel_to_point(depth_intrin, center, dis)
                    #data[1] = data[1] - 0.0425 #对齐
                    udp_send(data)
                elif count_result == 0.0 and data_pre != [0.0,0.0,0.0,0.0]:
                    data = [0.0,0.0,0.0,0.0]
                    udp_send(data)

                data_pre = data
                #print("camera_coordinate:",camera_coordinate)

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

    view_img = True

    # serial threading
    ser = serial_class('COM3', 115200)
    ser.receive_start()
    # model setting
    num_step = 10
    scale = 2500
    data_tensor = torch.empty((0,num_step,24))
    data_tensor_line = torch.empty((0,24))
    data_tensor_diff = torch.zeros((1,num_step,24))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    curr_prefac = np.float32(1/16)
    alpha = np.float32(np.exp(-1/16))
    beta = np.float32(1 - 1/16)
    load_model_name = "lsm_net_diff_yolo.pth"
    Win, Wlsm = initWeights1(LqWin=27, LqWlsm=2, in_conn_density=0.15, in_size=24, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None)
    lsm_net = LSM(1000, 24, 3, np.float32(np.zeros((1000,24))), np.float32(np.zeros((1000,1000))), alpha=alpha, beta=beta, th=20).to(device)
    lsm_net.load_state_dict(torch.load(load_model_name))
    lsm_net.eval()

    ten_results = []
    odor_result = ["", (255, 255, 255)]
    yolo_fruits = yolo_fruits_detect()
    to_pil_image = transforms.ToPILImage()
    # # udp threading
    # udp_thread = threading.Thread(target = udp_send)
    # udp_thread_running = True
    # udp_thread.start()

    while ser.receive_thread_running:
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

            if yolo_fruits.output is not None:
                im0 = yolo_fruits.output[0].plot()
            elif yolo_fruits.color_image is not None:
                im0 = yolo_fruits.color_image
            else:
                im0 = np.zeros((480, 640, 3))
            cv2.putText(im0, odor_result[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, odor_result[1], 2)
            cv2.imshow("detection", im0)

            if cv2.waitKey(1) == ord('q'):
                yolo_fruits.detect_running = False  # q to quit
                ser.receive_thread_running = False
                cv2.destroyAllWindows()
                break