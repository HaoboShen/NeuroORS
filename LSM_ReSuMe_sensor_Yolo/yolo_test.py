import cv2
import torch
import numpy as np
import pyrealsense2 as rs

from ultralytics import YOLO
def detect():

    # Load model
    model = YOLO('yolov5nu_fruits.pt')  # load modelpyt

    model.conf_thres = 0.9

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
        output = model(color_image,verbose=False)
        im0 = output[0].plot()
        # Stream results
        if view_img:
            cv2.imshow("detection", im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                break
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    view_img = True

    with torch.no_grad(): # 一个上下文管理器，被该语句wrap起来的部分将不会track梯度
        detect()
