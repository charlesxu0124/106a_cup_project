#!/usr/bin/python3
from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
import numpy as np
import time

import rospy
from wsg_50_common.msg import Cmd, Status

rospy.init_node('gripper', anonymous=True)

GRIPPER_CLOSE = 6   # chosen so that gripper closes entirely without pushing against itself
GRIPPER_OPEN = 96   # chosen so that gripper opens entirely without pushing against outer rail
ROS_NODE_TIMEOUT = 600     # kill script if waiting for more than 10 minutes on gripper
MAX_TIMEOUT = 10

def gripper_callback(status):
    # print(status)
    """ when working:
    status: "UNKNOWN"
    width: 96.00011444091797
    speed: -0.0006712675094604492
    acc: 0.0
    force: 0.0
    force_finger0: 0.0
    force_finger1: 0.0
    """

    desired_gpos = GRIPPER_CLOSE
    gripper_speed = 300

    cmd = Cmd()
    cmd.pos = desired_gpos
    cmd.speed = gripper_speed
    gripper_pub.publish(cmd)

gripper_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
rospy.Subscriber("/wsg_50_driver/status", Status, gripper_callback)

rospy.spin()