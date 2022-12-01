#!/usr/bin/python3
# from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv

# env = SawyerReachXYZEnv(
#     action_mode='position',
#     config_name='austri_config',
#     reset_free=False,
# 	position_action_scale=0.01,
# 	max_speed=0.4,
# )
# import ipdb; ipdb.set_trace()
# env.reset()

import rospy
from sawyer_control.srv import observation
import numpy as np

import argparse

import rospy

import intera_interface
import intera_external_devices
import pickle

from intera_interface import CHECK_VERSION

rospy.init_node('sawyer_env', anonymous=True)

limb = intera_interface.Limb()
joints = limb.joint_angles() # ["joints"]
print("joints")
print(joints)

pose = limb.endpoint_pose() # ["pose"]
print("pose")
print(pose)

point = pose['position']
end_eff_xyz = [point.x, point.y, point.z]
print("xyz")
print(end_eff_xyz)

quat = pose["orientation"]
end_eff_quat = [quat.x, quat.y, quat.z, quat.w]
print("quat")
print(end_eff_quat)

data = dict(
	joints=joints,
	end_eff_xyz=np.array(end_eff_xyz),
	end_eff_quat=np.array(end_eff_quat),
)

pickle.dump(data, open("post_iros2022_reset.p", "wb"))

# ('joints', {'right_j6': -4.5361787109375, 'right_j5': -1.47219140625, 'right_j4': -1.152875, 'right_j3': -1.333642578125, 'right_j2': -1.01491015625, 'right_j1': 1.0756220703125, 'right_j0': -2.5006318359375})
# ('pose', {'position': Point(x=-0.10526752452676857, y=-0.7327750783566074, z=-0.08670022173255966), 'orientation': Quaternion(x=0.49272195348547043, y=-0.5039775140207705, z=0.5235572057851725, w=-0.47866438574989933)})