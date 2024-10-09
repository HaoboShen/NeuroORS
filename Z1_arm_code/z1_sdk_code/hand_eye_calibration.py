# import z1 arm
import sys
sys.path.append("/home/shb/unitree_z1_ws/src/z1_sdk/lib")
import unitree_arm_interface

# import realsense
import pyrealsense2 as rs

# others
import cv2
import numpy as np
import time 
import math
import threading

class HandEyeCalibration():

    def __init__(self):
        super().__init__()
        # arm parameters
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=True)
        self.joint_speed = 2.0
        self.gripper_grip = 0.0
        
        # camera init
        self.frames = None
        self.depth_image = np.empty(0)
        self.color_image = np.empty(0)
        self.depth_colormap = None
        self.ret = None
        self.corners = None
        #self.corner_image = np.empty(0)
        self.image_threading = threading.Thread(target=self.image_thread)
        self.image_threading_flag = True
        self.image_threading.start()
        self.get_target_threading = threading.Thread(target=self.get_target)
        self.get_target_threading_flag = True
        self.get_target_threading.start()

        # chess board parameters
        self.chessboard_size = (6,9)
        self.square_size = 0.027

        # arm cal pos plan parameters
        self.angle_delta = 0.25
        self.translation_delta = 0.05
        self.planned_pos = []
        self.move_gap_time = 3 # 2s

        # label run to forward pos
        self.arm.loopOn()
        #print("before")
        self.arm.labelRun("forward")
        #self.arm.MoveJ(np.array([0.55820,0.98971,-1.64096,-0.18162,-0.18348,0.33395]),self.gripper_grip,self.joint_speed)

        self.arm.MoveJ(np.array([0.00542,1.17487,0.00705,0.46830,-0.00022,0.28166]),self.gripper_grip,self.joint_speed)
        #print("after")
        self.arm.loopOff()

        # calibration info
        # arm
        self.R_gripper2base = []
        self.T_gripper2base = []
        # camera
        self.R_target2cam = []
        self.T_target2cam = []

    def image_thread(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            #self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg = self.pipeline.start(self.config)
            profile = cfg.get_stream(rs.stream.color)
            self.intr = profile.as_video_stream_profile().get_intrinsics()
            print("camera intrinsics:",self.intr)
            self.cameraMatrix = np.zeros((3,3))
            self.cameraMatrix[0][0] = self.intr.fx
            self.cameraMatrix[1][1] = self.intr.fy
            self.cameraMatrix[0][2] = self.intr.ppx
            self.cameraMatrix[1][2] = self.intr.ppy
            self.coeffs = np.array(self.intr.coeffs)
            print("cameraMatrix:",self.cameraMatrix)
            print("cameracoeffs:",self.coeffs)
            while self.image_threading_flag:
                self.get_image()
               # if self.depth_image.size==0 or self.color_image.size==0:
                if self.color_image.size==0:
                    continue
                 # self.get_target()
                if self.ret:
                    self.color_image = cv2.drawChessboardCorners(self.color_image, self.chessboard_size, self.corners, self.ret)
                    cv2.imshow("corners",self.corner_image)
                #images = np.hstack((self.color_image,self.depth_colormap))
                images = self.color_image
                cv2.namedWindow("RealSense",cv2.WINDOW_AUTOSIZE)
                cv2.imshow("RealSense",images)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        except Exception as e:
            print("Error:",e)
        finally:
            self.pipeline.stop()
            print("camera end")

    def cal_pos_plan(self):
        # get current position
        curr_pos = self.arm.lowstate.endPosture
        eye_array = np.eye(6)
        # plan different positions
        for i in range(3):
            # change angle
            pos_tmp = curr_pos + eye_array[i]*self.angle_delta
            self.planned_pos.append(pos_tmp)
            
            pos_tmp = curr_pos - eye_array[i]*self.angle_delta
            self.planned_pos.append(pos_tmp)

        for i in range(3,6):
            # change translation

            pos_tmp = curr_pos + eye_array[i]*self.translation_delta
            self.planned_pos.append(pos_tmp)
            
            pos_tmp = curr_pos - eye_array[i]*self.translation_delta
            self.planned_pos.append(pos_tmp)

    def rpy2R(self, rpy): # [r,p,y] 单位rad
        rot_x = np.array([[1, 0, 0],
                        [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                        [0, math.sin(rpy[0]), math.cos(rpy[0])]])
        rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                        [0, 1, 0],
                        [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
        rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                        [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                        [0, 0, 1]])
        R = np.dot(rot_z, np.dot(rot_y, rot_x))
        return R

    def get_image(self):
        self.frames = self.pipeline.wait_for_frames()
        #self.depth_image = np.asanyarray(self.frames.get_depth_frame().get_data())
        self.color_image = np.asanyarray(self.frames.get_color_frame().get_data())
        #self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)

    def get_target(self):
        # self.get_image()
        while self.get_target_threading_flag:
            if self.color_image.size != 0:
                image = self.color_image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.ret, self.corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
                
                if self.ret:
                    tmp_pos = self.arm.lowstate.endPosture
                    print("***************************************************************************************************************")
                    print("get arm pos:",tmp_pos)
                    self.pos_update(tmp_pos)
                    self.corner_image = cv2.drawChessboardCorners(image, self.chessboard_size, self.corners, self.ret)
                    objp = np.zeros((np.prod(self.chessboard_size), 3), dtype=np.float32)
                    objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)*self.square_size
                    _, rvecs, tvecs = cv2.solvePnP(objp, self.corners.reshape(-1,2), self.cameraMatrix, self.coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    rot_mat,_ =cv2.Rodrigues(rvecs)
                    self.R_target2cam.append(rot_mat)
                    self.T_target2cam.append(tvecs)
                    print("rotmat:",rot_mat)
                    print("tvecs:",tvecs)
        print("target threading end")

    def pos_update(self,base2gripper):
        R_gripper2base = self.rpy2R(base2gripper[:3]).T
        T_gripper2base = np.dot(R_gripper2base,-base2gripper[3:])
        self.R_gripper2base.append(R_gripper2base)
        self.T_gripper2base.append(T_gripper2base)
        print("R:",R_gripper2base)
        #print("total R:",self.R_gripper2base)
        print("T:",T_gripper2base)
        #print("total T:",self.T_gripper2base)

    def calibrate(self):
        self.cal_pos_plan()
        self.arm.loopOn()
        for p in self.planned_pos:
            print("move to:",p)
            self.arm.MoveJ(p,self.gripper_grip,self.joint_speed)
            time.sleep(self.move_gap_time)
            #print("final_R:",np.array(self.R_gripper2base),"final_T:",np.array(self.T_gripper2base))
        self.arm.backToStart()
        self.arm.loopOff()
        self.get_target_threading_flag = False
        self.image_threading_flag = False
        print("plan pos number:",len(self.planned_pos))
        R_gripper2base = np.array(self.R_gripper2base)
        T_gripper2base = np.array(self.T_gripper2base)
        R_target2cam = np.array(self.R_target2cam)
        T_target2cam = np.array(self.T_target2cam)
        print("R_gripper2base:",R_gripper2base, R_gripper2base.shape)
        print("T_gripper2base:",T_gripper2base, T_gripper2base.shape)
        print("R_target2cam:",R_target2cam, R_target2cam.shape)
        print("T_target2cam:",T_target2cam, T_target2cam.shape)
        R,T=cv2.calibrateHandEye(R_gripper2base,T_gripper2base,R_target2cam,T_target2cam,method = cv2.CALIB_HAND_EYE_PARK)
        np.savez('RT.npz', R=R, T=T)
        print("result_R:",R)
        print("result_T:",T)
if __name__ == '__main__':
    hec = HandEyeCalibration()
    hec.calibrate()
