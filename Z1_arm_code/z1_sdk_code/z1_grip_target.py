import sys
sys.path.append("/home/shb/unitree_z1_ws/src/z1_sdk/lib")
import unitree_arm_interface
import time 
import numpy as np
import socket
class griptask():
    def __init__(self):
        
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=True) 
        self.joint_speed = 2.0
        self.gripper_grip = -0.5
        self.gripper_release = -1.5
        # targets
        self.target1_init_pos = np.array([])
        self.target1_grip_target_pos = np.array([])
        self.target1_release_pos = np.array([])

        self.target2_init_pos = np.array([])
        self.target2_grip_target_pos = np.array([])
        self.target2_release_pos = np.array([])


    def grip(self,init_pos,grip_target_pos,release_pos):
        
        self.arm.loopOn()
        self.arm.labelRun("forward")

        # pos

        # move to target and grip
        self.arm.MoveJ(init_pos, self.gripper_release, self.joint_speed)
        self.arm.MoveJ(grip_target_pos, self.gripper_release, self.joint_speed)
        print("move to target",grip_target_pos)

        self.arm.MoveJ(grip_target_pos, self.gripper_grip, self.joint_speed)
        print("grip target")

        # lift
        self.arm.MoveJ(init_pos, self.gripper_grip, self.joint_speed)
        print("lift")

        # move to destination and release
        self.arm.MoveJ(release_pos, self.gripper_grip, self.joint_speed)
        print("move to destination")

        self.arm.MoveJ(release_pos, self.gripper_release, self.joint_speed)
        print("release")

        self.arm.backToStart()
        #self.arm.loopOff()

    def get_rpy_from_xyz(self,x,y,z):
        '''
        roll = -1.5
        yaw = np.arctan2(y,x)
        a = np.sqrt(x**2+y**2+z**2)
        #print("a:",a)
        b = 0.35
        c = 0.23
        cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
        #print("cos_C:",cos_C)
        if cos_C>1 or cos_C<-1:
            pitch = 1.0
        else:
            pitch = np.arccos(cos_C)
        #if np.isnan(pitch):
        #    pitch = 0.785
        '''
        roll = -1.5
        pitch = 0.785
        yaw = np.arctan2(y,x)
        return roll,pitch,yaw

    def get_pos_from_xyz(self,xyz):
        roll,pitch,yaw = self.get_rpy_from_xyz(xyz[0],xyz[1],xyz[2])
        pos = np.concatenate((np.array([roll,pitch,yaw]).reshape(1,-1),xyz.reshape(1,-1)),axis = 1)[0]
        return pos

if __name__ == "__main__":
        # load RT
        with np.load('RT.npz') as data_file:
            R = data_file['R']
            T = data_file['T']
        #roll,pitch,yaw = -1.5,0.785,0
        print("R:",R,R.shape)
        print("T:",T,T.shape)
        # grip
        g = griptask()
        grip_finished = False
        # udp
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("192.168.85.130", 10000))
        # target label
        get_orange_flag = False
        get_peach_flag = False
        while True:
            data,address = s.recvfrom(65535)
            result = np.frombuffer(data,dtype=float)
            print(result)
        
            result = result.reshape(-1,4)
           # print(result)
            get_orange_flag = False
            get_peach_flag = False
            for r in result:
                if r[0] == 0.0:
                    grip_finished = False
                elif r[0] == 2.0:
                    peach_cam_pos = r[1:].reshape(3,1)
                    peach_cam_pos = peach_cam_pos + np.array([[0.02],[-0.03],[0.0]])
                    # peach_cam_pos[2]+=0.025
                    get_peach_flag = True
                    peach_xyz = R@peach_cam_pos+T
                    peach_pos = g.get_pos_from_xyz(peach_xyz[:,0])
                    print("peach_pos:",peach_pos,peach_pos.shape)
                    print("grip_finished:",grip_finished)
                    peach_pos_init_xyz = peach_xyz[:,0] + np.array([0.0,0.0,0.2])
                    peach_pos_init = g.get_pos_from_xyz(peach_pos_init_xyz)
                    #peach_pos_release_xyz = peach_xyz[:,0] - np.array([0.1,0.1,0])
                    #peach_pos_release_xyz = np.array([0.1,-0.26,0.0])
                    #peach_pos_release = g.get_pos_from_xyz(peach_pos_release_xyz)
                    peach_pos_release = np.array([-1.5,0.0,-1.5,0.0,-0.3,0.4])
                elif r[0] == 1.0:
                    orange_cam_pos = r[1:].reshape(3,1)
                    orange_cam_pos = orange_cam_pos + np.array([[0.02],[-0.045],[0.0]])
                    get_orange_flag  = True
                    orange_xyz = R@orange_cam_pos+T
                    #print("orange_cam:",orange_cam_pos,orange_cam_pos.shape)
                    orange_pos = g.get_pos_from_xyz(orange_xyz[:,0])
                    print("orange_pos:",orange_pos,orange_pos.shape)
                    print("grip_finished:",grip_finished)
                    orange_pos_init_xyz = orange_xyz[:,0] + np.array([0.0,0.0,0.2])
                    orange_pos_init = g.get_pos_from_xyz(orange_pos_init_xyz)
                    #orange_pos_release_xyz = orange_xyz[:,0] - np.array([0.1,0.1,0])
                    #orange_pos_release_xyz = np.array([0.1,-0.26,0.0])
                    #orange_pos_release = g.get_pos_from_xyz(orange_pos_release_xyz)
                    orange_pos_release = np.array([-1.5,0.0,-1.5,0.0,-0.3,0.4])
            if not grip_finished:
                if get_orange_flag:
                    g.grip(orange_pos_init,orange_pos,orange_pos_release)
                    grip_finished = True

                if get_peach_flag:
                    g.grip(peach_pos_init,peach_pos,peach_pos_release)
                    grip_finished = True
            
