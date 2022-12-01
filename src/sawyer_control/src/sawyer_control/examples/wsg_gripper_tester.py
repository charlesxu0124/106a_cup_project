#!/usr/bin/python3
from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
import numpy as np
import time

import rospy
from wsg_50_common.msg import Cmd, Status

from sawyer_control.grippers.wsg_gripper import WSG50Gripper

rospy.init_node('gripper', anonymous=True)

gripper = WSG50Gripper()

gripper.do_open_cmd()
time.sleep(2)
gripper.do_close_cmd()
time.sleep(2)
gripper.do_open_cmd()
time.sleep(2)
