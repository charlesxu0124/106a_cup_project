from geometry_msgs.msg import Quaternion
import numpy as np
#JOINT_CONTROLLER_SETTINGS
JOINT_POSITION_SPEED = .1
JOINT_POSITION_TIMEOUT = .2

#JOINT INFO
JOINT_NAMES = ['right_j0',
               'right_j1',
               'right_j2',
               'right_j3',
               'right_j4',
               'right_j5',
               'right_j6'
               ]
LINK_NAMES = ['right_l2', 'right_l3', 'right_l4', 'right_l5', 'right_l6', '_hand']


# RESET_ANGLES = np.array(
#     [0.15262207,  0.91435349, -2.02594233,  1.6647979, -2.41721773, 1.14999604, -2.47703505]
# )

# RESET_DICT = dict(zip(JOINT_NAMES, RESET_ANGLES))
# POSITION_CONTROL_EE_ORIENTATION=Quaternion(
#     x=0.72693193, y=-0.03049006, z=0.6855942, w=-0.02451418
# )

RESET_DICT = {'right_j6': -4.5361787109375, 'right_j5': -1.47219140625, 'right_j4': -1.152875, 'right_j3': -1.333642578125, 'right_j2': -1.01491015625, 'right_j1': 1.0756220703125, 'right_j0': -2.5006318359375}
POSITION_CONTROL_EE_ORIENTATION=Quaternion(
    x=0.49272195348547043, y=-0.5039775140207705, z=0.5235572057851725, w=-0.47866438574989933
)
