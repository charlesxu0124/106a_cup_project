from sawyer_control.configs.base_config import *
from geometry_msgs.msg import Quaternion
import numpy as np
from gym.spaces import Box
import json
import os

## Run env_calibration.py to generate json file ###

# load config data from json filed
#****************************************************************************************************************
dirname = os.path.abspath(os.path.dirname(__file__))
sawyer_blocks_config = os.path.join(dirname, 'env_calibration.json')

if os.path.isfile(sawyer_blocks_config):
    with open(sawyer_blocks_config, 'r') as json_file:
        json_data = json.load(json_file)
        if 'endeffPose_goal' in json_data:
            endeffPose_goal_loaded = json_data['endeffPose_goal']
            endeffPose_goal_loaded_successfully = True
            print("endeffPose_goal_loaded = ", endeffPose_goal_loaded)
        else:
            endeffPose_goal_loaded_successfully = False

        if 'endeffPose_reset' in json_data:
            endeffPose_reset_loaded = json_data['endeffPose_reset']
            endeffPose_reset_loaded_successfully = True
            print("endeffPose_reset = ", endeffPose_reset_loaded)
        else:
            endeffPose_reset_loaded_successfully = False

        if 'jointAngles_reset' in json_data:
            jointAngles_loaded = json_data['jointAngles_reset']
            jointAngles_loaded_successfully = True
            print("jointAngles_loaded = ", jointAngles_loaded)
        else:
            jointAngles_loaded_successfully = False

        if 'Quaternions_goal' in json_data:
            if len(json_data['Quaternions_goal']) == 4:
                Quaternions_goal_loaded = json_data['Quaternions_goal']
                Quaternions_goal_loaded_successfully = True
                print("Quaternions_goal_loaded = ", Quaternions_goal_loaded)
            else: 
                Quaternions_goal_loaded_successfully = False
        else:
            Quaternions_goal_loaded_successfully = False

        if all (k in json_data for k in ('blockPoseID0', 'blockOrientationID0', 'blockPoseID1',
                                'blockOrientationID1', 'blockPoseID2', 'blockOrientationID2')):
            des_pose_block_id0_loaded = np.array(json_data['blockPoseID0']+json_data['blockOrientationID0'])
            des_pose_block_id1_loaded = np.array(json_data['blockPoseID1']+json_data['blockOrientationID1'])
            des_pose_block_id2_loaded = np.array(json_data['blockPoseID2']+json_data['blockOrientationID2'])
            blocks_loaded_successfully = True
            print("des_pose_block_id0_loaded = ", des_pose_block_id0_loaded)
            print("Blocks loaded successfully")
        else:
            blocks_loaded_successfully = False

else:
    endeffPose_goal_loaded_successfully = False
    endeffPose_reset_loaded_successfully = False
    jointAngles_loaded_successfully = False
    Quaternions_goal_loaded_successfully = False
    blocks_loaded_successfully = False

# default values
if not endeffPose_goal_loaded_successfully:
    print("WARNING: sawyer_blocks_config.json does not contain endeffPose_goal")
    endeffPose_goal_loaded = (0.533, 0.255, 0.207) 
if not endeffPose_reset_loaded_successfully:
    print("WARNING: sawyer_blocks_config.json does not contain endeffPose_reset")
    endeffPose_reset_loaded =  np.array([0.532, 0.223, 0.285]) 
if not jointAngles_loaded_successfully:
    print("WARNING: sawyer_blocks_config.json does not contain jointAngles_reset")
    jointAngles_loaded = [0.31754395, -1.02691996, -0.29332128, 1.79010248, 0.23310059, 0.83045018, 3.13426757]
if not Quaternions_goal_loaded_successfully:
    print("WARNING: sawyer_blocks_config.json does not contain Quaternions_goal")
    Quaternions_goal_loaded =  np.array([0.718, 0.718, 0, 0]) 
if not blocks_loaded_successfully:
    print("WARNING: sawyer_blocks_config.json does not contain blocks")
    des_pose_block_id0_loaded = np.array([ 0.16 , -0.04,  0.01,  1.912,  0.057, -1.55])
    des_pose_block_id1_loaded = np.array([0.16, 0.038, 0.015, 1.90,  0.086, -1.68])
    des_pose_block_id2_loaded = np.array([0.16, -0.00143171,  0.01311876, 1.774 , -0.018, -1.644])

# ****************************************************************************************************************


POSITION_GOAL_POS = endeffPose_goal_loaded 
POSITION_RESET_POS = endeffPose_reset_loaded
RESET_ANGLES = np.array(jointAngles_loaded)
RESET_DICT = dict(zip(JOINT_NAMES, RESET_ANGLES))

QUATERNIONS_GOAL = Quaternions_goal_loaded

POSITION_CONTROL_EE_ORIENTATION = Quaternion(
    x=QUATERNIONS_GOAL[0],
    y=QUATERNIONS_GOAL[1],
    z=QUATERNIONS_GOAL[2],
    w=QUATERNIONS_GOAL[3],
    )

#### NEW RESET POSITION AND ORIENTATION!

import pickle
data = pickle.load(open("/home/tesla/ros_ws/reset.p", "rb"), encoding="bytes")
POSITION_CONTROL_EE_ORIENTATION = data[b"end_eff_quat"] # np.array([0.014567714859877588, 0.704467276754497, -0.0010676501442130257, 0.7095861454319938])
POSITION_RESET_POS = data[b"end_eff_xyz"] # np.array([0.6380148574086548, -0.037770734735271094, 0.01143441775195979])

QUATERNIONS_GOAL = POSITION_CONTROL_EE_ORIENTATION

pos_low = np.zeros_like(POSITION_GOAL_POS)
pos_high = np.zeros_like(POSITION_GOAL_POS)

max_offset = 0.1 # 10cm

# pos_low[0] = -0.28 # POSITION_RESET_POS[0] - max_offset
# pos_high[0] = 0.1 # POSITION_RESET_POS[0] + 2 * max_offset
# pos_low[1] = POSITION_RESET_POS[1]#0.3 # POSITION_RESET_POS[1] - max_offset
# pos_high[1] = 0.74 # POSITION_RESET_POS[1] + max_offset
# pos_low[2] = 0.14 # POSITION_RESET_POS[2] - 0.02 # max_offset
# pos_high[2] = 0.16 # POSITION_RESET_POS[2] + 0.02 # max_offset

### DEFAULT (Task 1 and 3)
# pos_low[0] = 0.45 #0.4 #0.5 # POSITION_RESET_POS[0] - max_offset
# pos_high[0] = 0.65 #0.775 #0.7 # POSITION_RESET_POS[0] + 2 * max_offset
# pos_low[1] = -0.25 #-0.115 #-0.2 #0.3 # POSITION_RESET_POS[1] - max_offset
# pos_high[1] = -0.025 #0.11 #0.24 # changed to 0.3 on 12/5/21 # POSITION_RESET_POS[1] + max_offset
# pos_low[2] = 0.14 # POSITION_RESET_POS[2] - 0.02 # max_offset
# pos_high[2] = 0.25 # POSITION_RESET_POS[2] + 0.02 # max_offset
pos_low[0] = 0.45 #0.4 #0.5 # POSITION_RESET_POS[0] - max_offset
pos_high[0] = 0.85 #0.775 #0.7 # POSITION_RESET_POS[0] + 2 * max_offset
pos_low[1] = -0.25 #-0.115 #-0.2 #0.3 # POSITION_RESET_POS[1] - max_offset
pos_high[1] = 0.3 #0.11 #0.24 # changed to 0.3 on 12/5/21 # POSITION_RESET_POS[1] + max_offset
pos_low[2] = 0.14 # POSITION_RESET_POS[2] - 0.02 # max_offset
pos_high[2] = 0.35 # POSITION_RESET_POS[2] + 0.02 # max_offset

## TODO(patrick) Task 2
# pos_low[0] = 0.4 #PATRICK HARDCODE, original=0.4 # POSITION_RESET_POS[0] - max_offset
# pos_high[0] = 0.8 # POSITION_RESET_POS[0] + 2 * max_offset
# pos_low[1] = -0.2 #0.3 # POSITION_RESET_POS[1] - max_offset
# pos_high[1] = 0.3 #0.24 # changed to 0.3 on 12/5/21 # POSITION_RESET_POS[1] + max_offset
# pos_low[2] = 0.14 # POSITION_RESET_POS[2] - 0.02 # max_offset
# pos_high[2] = 0.25 #PATRICK HARDCODE, original=0.4 # POSITION_RESET_POS[2] + 0.02 # max_offset

POSITION_SAFETY_BOX_LOWS = pos_low
POSITION_SAFETY_BOX_HIGHS = pos_high
POSITION_SAFETY_BOX = Box(POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS)

ORIENTATION_RESET = np.array(QUATERNIONS_GOAL)

BLOCK_POSE_ID1 = des_pose_block_id0_loaded
BLOCK_POSE_ID2 = des_pose_block_id1_loaded
BLOCK_POSE_ID3 = des_pose_block_id2_loaded

END_EFFECTOR_POS_LOW = -1.2 * np.ones(3)
END_EFFECTOR_POS_HIGH = 1.2 *np.ones(3)

END_EFFECTOR_VEL_LOW = -2 * np.ones(6)
END_EFFECTOR_VEL_HIGH = 2 *np.ones(6)

END_EFFECTOR_ANGLE_LOW = -1*np.ones(4)
END_EFFECTOR_ANGLE_HIGH = np.ones(4)

END_EFFECTOR_FORCE_LOW = -20*np.ones(1)
END_EFFECTOR_FORCE_HIGH = 20*np.ones(1)

END_EFFECTOR_TORQUE_LOW = -1*np.ones(3)
END_EFFECTOR_TORQUE_HIGH = 1*np.ones(3)

END_EFFECTOR_VALUE_LOW = {
    'position': END_EFFECTOR_POS_LOW,
    'velocity': END_EFFECTOR_VEL_LOW,
    'angle': END_EFFECTOR_ANGLE_LOW,
    'force': END_EFFECTOR_FORCE_LOW,
    'torque': END_EFFECTOR_TORQUE_LOW,
}

END_EFFECTOR_VALUE_HIGH = {
    'position': END_EFFECTOR_POS_HIGH,
    'velocity': END_EFFECTOR_VEL_HIGH,
    'angle': END_EFFECTOR_ANGLE_HIGH,
    'force': END_EFFECTOR_FORCE_HIGH,
    'torque': END_EFFECTOR_TORQUE_HIGH,
}


