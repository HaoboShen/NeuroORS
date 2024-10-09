import socket
import threading
import time
import cv2
import torch
import numpy as np
import pyrealsense2 as rs

from ultralytics import YOLO
def detect():
    global camera_coordinate
    # Load model
    model = YOLO('yolov8n_fruits.pt')  # load modelpyt

    pipeline = rs.pipeline()
    # 创建 config 对象：
    config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    # Start streaming
    pipeline.start(config)
    align_to_color = rs.align(rs.stream.color)
    while True:
        # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)
        # depth_frame = frames.get_depth_frame()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())


        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

        output = model(color_image,verbose=False)
        im0 = output[0].plot()
        boxes = output[0].boxes
        print("boxes:",boxes)
        camera_coordinate = []
        for i,cls in enumerate(boxes.cls):
            
            # if cls == 14: #13: orange 14: peach 5: apple
            center = [(boxes.xyxy[i][0]+boxes.xyxy[i][2]).item()/2,(boxes.xyxy[i][1]+boxes.xyxy[i][3]).item()/2]
            print("center:",center,cls.item())

            dis = depth_frame.get_distance(int(center[0]),int(center[1]) )  # 获取该像素点对应的深度
            camera_coordinate.append([cls.item()]+rs.rs2_deproject_pixel_to_point(depth_intrin, center, dis))
            print("camera_coordinate:",camera_coordinate)
        
        # Stream results
        if view_img:
            cv2.imshow("detection", im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                break
    cv2.destroyAllWindows()
    
def udp_send():
    global count_result
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('192.168.85.130', 10000)
    # server_socket.bind(server_address)
    while udp_thread_running:
        server_socket.sendto(np.array(camera_coordinate).tobytes(), server_address)
        time.sleep(1)
    server_socket.close()

if __name__ == '__main__':

    view_img = True
# udp threading
    udp_thread = threading.Thread(target = udp_send)
    udp_thread_running = True
    udp_thread.start()
    camera_coordinate = []
    with torch.no_grad(): 
        detect()
