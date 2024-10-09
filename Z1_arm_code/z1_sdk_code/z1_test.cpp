#include "unitree_arm_sdk/control/unitreeArm.h"

//#include <unistd.h>
int main()
{
	UNITREE_ARM::unitreeArm arm(true);
	arm.sendRecvThread->start();
	//arm.backToStart();
	
	Vec6 grip_target_pos,release_pos,init_pos; 
	grip_target_pos << 1.74706,1.46363,-1.18069,0.08343,-0.25496,0.08403;
	release_pos << 1.72512,1.08527,-1.20144,0.08214,-0.46656,0.15109;
	init_pos << 1.76201,1.47742,-1.16642,0.08286,-0.25358,0.16467;
	double joint_speed = 2.0;
	double gripper_grip = -0.5;
	double gripper_release = -1.5;
	
	//move to initPos and release the gripper
	//arm.labelRun("forward");
	std::cout<<"move to initpos"<<arm.lowstate->endPosture.transpose()<<std::endl;
	/*
	//move to target and grip	
	arm.MoveJ(init_pos, gripper_release, joint_speed);
	arm.MoveJ(grip_target_pos, gripper_release, joint_speed);
	std::cout<<"move to target"<<std::endl;

	arm.MoveJ(grip_target_pos, gripper_grip, joint_speed);
	std::cout<<"grip target"<<std::endl;
	//lift
	arm.MoveJ(init_pos, gripper_grip, joint_speed);
	std::cout<<"lift"<<std::endl;
	//move to destination and release	
	arm.MoveJ(release_pos, gripper_grip, joint_speed);
	std::cout<<"move to destination"<<std::endl;

	arm.MoveJ(release_pos, gripper_release, joint_speed);
	std::cout<<"release"<<std::endl;

	arm.backToStart();
	*/
	arm.sendRecvThread->shutdown();

}
