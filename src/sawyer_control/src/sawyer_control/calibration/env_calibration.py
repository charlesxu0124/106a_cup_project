# # #!/usr/bin/python3
import sawyer_control.envs.sawyer_reaching
import numpy as np
from termcolor import colored
import time
import json
import os
import rospy

dirname = os.path.abspath(os.path.dirname(__file__))
json_file = os.path.abspath(os.path.join(dirname, '../configs/env_calibration.json'))
json_file_save = os.path.abspath(os.path.join(dirname, '../configs/env_calibration_save.json'))

if os.path.isfile(json_file):  
	with open(json_file, 'r', encoding='utf-8') as read_data:
			data = json.load(read_data)
else:
	data = {}

with open(json_file_save, 'w', encoding='utf-8') as outfile:
	json.dump(data, outfile, ensure_ascii=False, indent=4)
print(colored('Created a copy of the current config at','green')) 
print(json_file_save)

env = sawyer_control.envs.sawyer_reaching.SawyerReachXYZEnv(action_mode='position', config_name = 'calibration_config', max_speed=0.025)

n = 1
useARtags = False


while True:
    qr = input('Please launch exp_nodes1 in another terminal. Is it running? [y/n] (ENTER = y)')
    if qr == '':
    	qr = 'y'
    	break
    elif not qr[0].lower() in ['y']:print(colored('Please turn on exp_nodes or exp_nodes1.', 'red'))
    else:break

if qr[0].lower() == 'y':
	print(colored('Move the robot arm to the desired goal location.', 'green')) 


while True:
	qr = input('Do you want to do the calibration without AR-tags? (otherwise AR-tag tracking must be running) [y/n] (ENTER = y)')
	if qr == '':
		qr = 'y'
		break
	elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))  
	else:break
if qr[0].lower() == 'n': 
	useARtags = True
if qr[0].lower() == 'y':
	pass

while True:
	qr = input('Do you want to use single measurements? (otherwise n = 300 measurements are averaged) [y/n] (ENTER = y)')
	if qr == '':
		qr = 'y'
		break
	elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))
	else:break
if qr[0].lower() == 'n': 
	n = 300
if qr[0].lower() == 'y':
	pass

if useARtags:
	print(colored("If the program freezes or crashes here: Make sure that the camera service, the AR-tracking, and exp_nodes are running in other terminals.", 'green'))
else:
	print(colored("If the program freezes or crashes here: Make sure that exp_nodes is running in another terminal.", 'green'))

endeffPose = np.zeros_like(env._get_endeffector_pose()[:3])
endeffQrientation = np.zeros_like(env._get_endeffector_pose()[3:])
jointAngles = np.zeros_like(env._get_joint_angles())

if useARtags:
	blockPoseID0 = np.zeros_like(env._get_ar_pos()['middle_marker_pose'][:3])
	blockOrientationID0 = np.zeros_like(env._get_ar_pos()['middle_marker_pose'][7:10])
	blockPoseID1 = np.zeros_like(env._get_ar_pos()['left_marker_pose'][:3])
	blockOrientationID1 = np.zeros_like(env._get_ar_pos()['left_marker_pose'][7:10])
	blockPoseID2 = np.zeros_like(env._get_ar_pos()['right_marker_pose'][:3])
	blockOrientationID2 = np.zeros_like(env._get_ar_pos()['right_marker_pose'][7:10])

# averaging measurements
for i in range(n):
	endeffPose = endeffPose + env._get_endeffector_pose()[:3]
	endeffQrientation = endeffQrientation + env._get_endeffector_pose()[3:]
	jointAngles = jointAngles + env._get_joint_angles()
	if useARtags:
		blockPoseID0 = blockPoseID0 + env._get_ar_pos()['middle_marker_pose'][:3]
		blockOrientationID0 = blockOrientationID0 + env._get_ar_pos()['middle_marker_pose'][7:10]
		blockPoseID1 = blockPoseID1 + env._get_ar_pos()['left_marker_pose'][:3]
		blockOrientationID1 = blockOrientationID1 + env._get_ar_pos()['left_marker_pose'][7:10]
		blockPoseID2 = blockPoseID2 + env._get_ar_pos()['right_marker_pose'][:3]
		blockOrientationID2 = blockOrientationID2 + env._get_ar_pos()['right_marker_pose'][7:10]
	time.sleep(0.005)
	if i%50 == 0 and i != 0:
		print('50 measurements done')

endeffPose = endeffPose/n
endeffQrientation = endeffQrientation/n
jointAngles = jointAngles/n
if useARtags:
	blockPoseID0 = blockPoseID0/n
	blockOrientationID0 = blockOrientationID0/n
	blockPoseID1 = blockPoseID1/n
	blockOrientationID1 = blockOrientationID1/n
	blockPoseID2 = blockPoseID2/n
	blockOrientationID2 = blockOrientationID2/n

while True:
	qr = input('Would you like to print the measured data? [y/n] (ENTER = y)')
	if qr == '':
		qr = 'y'
		break
	elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))
	else:break
if qr[0].lower() == 'y': 
	print(" ")
	print(colored('Current endeffector position expressed in 3D-Space-coordinates', 'red'))
	print("After moving the robot to its goal or reset position, use the following coordinates to replace 'fixed_goal' or 'reset_pos' in 'sawyer_control/src/sawyer_control/envs/sawyer_insertion.py' ")
	print(colored('x, y, z: ', 'green'), colored(repr(endeffPose), 'green')) # x,y,z
	print(" ")
	print(colored('Current endeffector pose expressed in quaternions', 'red'))
	print("After moving the robot to its goal position (not reset position), use the following pose to replace 'Q' in 'sawyer_control/scripts/ik_server.py' ")
	print(colored('x, y, z, w: ', 'green'), colored(repr(endeffQrientation), 'green')) # x,y,z,w
	print(" ")
	print(colored('Current joint angles expressed in joint-space coordinates', 'red'))
	print("After moving the robot to its goal or reset position, use the following joint angles to replace 'tgt_jnt_angles' in 'sawyer_control/scripts/joint_space_impd_subscriber.py' or 'RESET_ANGLES' in 'sawyer_control/src/sawyer_control/configs/ros_config.py'")
	print(colored(repr(jointAngles), 'green'))

	if useARtags:
		# Blocks Positions and Orientations
		print(" ")
		print(" ")
		print(" ")
		print(" ")
		print(colored('Block ID0: desired block position expressed in 3D-Space-coordinates and block orientation expressed in Euler-angles', 'red'))
		print("Put the block into its desired position/orientation and use the following printed values to replace 'des_pose_block_id0' in 'sawyer_control/src/sawyer_control/envs/sawyer_insertion.py' ")
		print(colored('x, y, z: ', 'green'), colored(repr(blockPoseID0), 'green')) # entries [3:] are the pose in quat
		print(colored('alpha [rad], beta [rad], gamma [rad]: ', 'green'), colored(repr(blockOrientationID0), 'green'))
		print(" ")
		print(colored('Block ID1: desired block position expressed in 3D-Space-coordinates and block orientation expressed in Euler-angles', 'red'))
		print("Put the block into its desired position/orientation and use the following printed values to replace 'des_pose_block_id1' in 'sawyer_control/src/sawyer_control/envs/sawyer_insertion.py' ")
		print(colored('x, y, z: ', 'green'), colored(repr(blockPoseID1), 'green')) # entries [3:] are the pose in quat
		print(colored('alpha [rad], beta [rad], gamma [rad]: ', 'green'), colored(repr(blockOrientationID1), 'green'))
		print(" ")
		print(colored('Block ID2: desired block position expressed in 3D-Space-coordinates and block orientation expressed in Euler-angles', 'red'))
		print("Put the block into its desired position/orientation and use the following printed values to replace 'des_pose_block_id2' in 'sawyer_control/src/sawyer_control/envs/sawyer_insertion.py' ")
		print(colored('x, y, z: ', 'green'), colored(repr(blockPoseID2), 'green')) # entries [3:] are the pose in quat
		print(colored('alpha [rad], beta [rad], gamma [rad]: ', 'green'), colored(repr(blockOrientationID2), 'green'))
		print(" ")

if qr[0].lower() == 'n':
	print("NEXT")
	pass

while True:
	qr = input('Would you like to save the measurements for the goal position? [y/n] (ENTER = y)')
	if qr == '':
		qr = 'y'
		break
	elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))
	else:break
if qr[0].lower() == 'y':


	if os.path.isfile(json_file):  
		with open(json_file, 'r', encoding='utf-8') as read_data:
			data = json.load(read_data)
	else:
		data = {}

	data['endeffPose_goal'] = endeffPose.tolist() 
	data['Quaternions_goal'] = endeffQrientation.tolist() 
	data['jointAngles_goal'] = jointAngles.tolist()

	with open(json_file, 'w', encoding='utf-8') as outfile:
		json.dump(data, outfile, ensure_ascii=False, indent=4)
	print("DONE")

if qr[0].lower() == 'n':
	print("NEXT")
	pass

if useARtags:
	while True:
		qr = input('Would you like to save the current block positions and orientations? [y/n] (ENTER = y)')
		if qr == '':
			qr = 'y'
			break
		elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))
		else:break
	if qr[0].lower() == 'y':

		if os.path.isfile(json_file):  
			with open(json_file, 'r', encoding='utf-8') as read_data:
				data = json.load(read_data)
		else:
			data = {}

		data['blockPoseID0'] = blockPoseID0.tolist()
		data['blockOrientationID0'] = blockOrientationID0.tolist()  
		data['blockPoseID1'] = blockPoseID1.tolist() 
		data['blockOrientationID1'] = blockOrientationID1.tolist()  
		data['blockPoseID2'] = blockPoseID2.tolist() 
		data['blockOrientationID2'] = blockOrientationID2.tolist()  

		with open(json_file, 'w', encoding='utf-8') as outfile:
			json.dump(data, outfile, ensure_ascii=False, indent=4)
		print("DONE")


	if qr[0].lower() == 'n':
		print("NEXT")
		pass

while True:
	print(colored('Answer NO if you want to set the reset position manually!', 'red'))
	qr2 = input('Would you like to use automated movement to set the reset position? [y/n] (ENTER = y)')
	if qr2 == '':
		qr2 = 'y'
		break
	elif not qr2[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))
	else:break

if qr2[0].lower() == 'y':
	print(colored('RESTARTING THE ENVIRONMENT TO UPDATE THE EE-ORIENTATION', 'red'))
	move_again = True
	counter = 0
	while move_again:
		
		qr = input('Please provide the relative reset pos height in cm: [0, 20] (ENTER = CONTINUE)')

		if not qr == '': 
			if str(qr[0]).lower() in ['y','n']:
				print('Please provide a value between 0 and 20cm') 
				continue
		if qr == '' or not 0 <= float(qr) <= 20:
				if qr == '' and not counter == 0:
					print('CONTINUE')
					move_again = False
				else:
					print('Please provide a value between 0 and 20cm')

		else:
			rel_reset_height = float(qr) / 100

			print(colored('Moving the robot arm to the reset position, {:s}cm above goal position.'.format(qr),  'green'))
			rospy.sleep(1)

			reset_pos = np.array(endeffPose) + np.array([0, 0, rel_reset_height])
			env.reach_goal_with_tol(reset_pos, tol = 0.0001, t = 12, orientation = endeffQrientation)

			counter += 1

if qr2[0].lower() == 'n':
	print("Automatic movement to reset position was skipped")
	print(colored('Manually move the robot arm to the reset position.', 'green'))
	pass


while True:
	qr = input('Would you like to measure the reset pose? [y/n] (ENTER = y)')
	if qr == '':
		qr = 'y'
		break
	elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))
	else:break
if qr[0].lower() == 'y':
	endeffPose = env._get_endeffector_pose()[:3]
	endeffQrientation = env._get_endeffector_pose()[3:]
	jointAngles = env._get_joint_angles()
	print('DONE')
	while True:
		qr = input('Would you like to print the measured data? [y/n] (ENTER = y)')
		if qr == '':
			qr = 'y'
			break
		elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))
		else:break
	if qr[0].lower() == 'y': 
		print(" ")
		print(colored('Current endeffector position expressed in 3D-Space-coordinates', 'red'))
		print("After moving the robot to its goal or reset position, use the following coordinates to replace 'fixed_goal' or 'reset_pos' in 'sawyer_control/src/sawyer_control/envs/sawyer_insertion.py' ")
		print(colored('x, y, z: ', 'green'), colored(repr(endeffPose), 'green')) # x,y,z
		print(" ")
		print(colored('Current endeffector pose expressed in quaternions', 'red'))
		print("After moving the robot to its goal position (not reset position), use the following pose to replace 'Q' in 'sawyer_control/scripts/ik_server.py' ")
		print(colored('x, y, z, w: ', 'green'), colored(repr(endeffQrientation), 'green')) # x,y,z,w
		print(" ")
		print(colored('Current joint angles expressed in joint-space coordinates', 'red'))
		print("After moving the robot to its goal or reset position, use the following joint angles to replace 'tgt_jnt_angles' in 'sawyer_control/scripts/joint_space_impd_subscriber.py' or 'RESET_ANGLES' in 'sawyer_control/src/sawyer_control/configs/ros_config.py'")
		print(colored(repr(jointAngles), 'green'))

	if qr[0].lower() == 'n':
		print("NEXT")
		pass

	while True:
		qr = input('Would you like to save the reset pose? [y/n] (ENTER = y)')
		if qr == '':
			qr = 'y'
			break
		elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red')) 
		else:break
	if qr[0].lower() == 'y' or qr == '':

		if os.path.isfile(json_file):
			with open(json_file, 'r', encoding='utf-8') as read_data:
				data = json.load(read_data)
		else:
			data = {}

		data['endeffPose_reset'] = endeffPose.tolist() 
		data['Quaternions_reset'] = endeffQrientation.tolist() 
		data['jointAngles_reset'] = jointAngles.tolist()

		with open(json_file, 'w', encoding='utf-8') as outfile:
			json.dump(data, outfile, ensure_ascii=False, indent=4)
		print("DONE")

	if qr[0].lower() == 'n':
		print("NEXT")
		pass


if qr[0].lower() == 'n':
	print("NEXT")
	pass


if False: 
	#This function has been removed due to safety concerns. Feel free to edit the file and test it.
	while True:
		qr = input('Would you like to set the reset and goal quaternions to [0.7071, 0.7071, 0, 0] (pointing down)? [y/n] (ENTER = y')
		if qr == '':
			qr = 'y'
			break
		elif not qr[0].lower() in ['y','n']:print(colored('Please enter again', 'red'))
		else:break
	if qr[0].lower() == 'y':
		print(colored('This function has been removed due to safety concerns. Feel free to edit the file and test it.'))
		if False:	
			endeffQrientation = np.array([0.70710678118, 0.70710678118, 0.0, 0.0]) 

			if os.path.isfile(json_file):  
				with open(json_file, 'r', encoding='utf-8') as read_data:
					data = json.load(read_data)
			else:
				data = {}

			data['Quaternions_reset'] = endeffQrientation.tolist() 
			data['Quaternions_goal']  = endeffQrientation.tolist() 

			with open(json_file, 'w', encoding='utf-8') as outfile:
				json.dump(data, outfile, ensure_ascii=False, indent=4)
			print("DONE")

	if qr[0].lower() == 'n':
		print("NEXT")
		pass

print(colored("DONE WITH CALIBRATION",'green'))
print('Everything you saved is stored in:', json_file)