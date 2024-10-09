import sys
sys.path.append("/home/shb/unitree_z1_ws/src/z1_sdk/lib")
import unitree_arm_interface
import time 
import numpy as np
class griptask():
    def __init__(self):
        
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=True) 
        self.joint_speed = 0.5
        self.gripper_grip = -0.5
        self.gripper_release = -1.5

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
        self.arm.loopOff()
if __name__ == "__main__":
        target = np.array([0,0,0.2])
        with np.load('RT.npz') as data_file:
            R = data_file['R']
            T = data_file['T']
        pos = np.dot(R,target.T)+T[:,0]
        print("R:",R)
        print("T:",T)
        print("first:",np.dot(R,target.T))
        print("pos:",pos)
        x,y,z = 0,0,0
        roll,pitch,yaw = 0,1.5,-1.0
        init_pos = np.array([roll,pitch,yaw,x,y,z+0.1])
        grip_target_pos = np.array([roll,pitch,yaw,x,y,z])
        release_pos = np.array([roll,pitch,yaw,x+0.1,y,z])
        #print("pos:",pos)
        '''
        #init_pos = np.array([1.76201,1.47742,-1.16642,0.08286,-0.25358,0.16467])
        #grip_target_pos = np.array([1.74706,1.46363,-1.18069,0.08343,-0.25496,0.08403])
        #release_pos = np.array([1.72512,1.08527,-1.20144,0.08214,-0.46656,0.15109])
        g = griptask()
        g.grip(init_pos,grip_target_pos,release_pos)
        '''
