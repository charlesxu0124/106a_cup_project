#!/usr/bin/env python
import intera_interface as ii
from joint_position_impedance_controller import PositionController
from sawyer_control.srv import *
import rospy

def execute_action(action_msg):
    action = action_msg.angles
    joint_names = arm.joint_names()
    joint_to_values = dict(zip(joint_names, action))
    duration = action_msg.duration
    ja = [joint_to_values[name] for name in arm.joint_names()]
    controller.move_along_spline_in_vel_mode([ja], duration=duration)
    return angle_actionResponse(True)

def angle_action_server():
    rospy.init_node('angle_action_server2', anonymous=True)
    global arm
    global controller
    arm = ii.Limb('right')
    arm.set_joint_position_speed(0.1)
    controller = PositionController(control_rate=1000)
    s = rospy.Service('angle_action2', angle_action, execute_action)
    rospy.spin()

if __name__ == '__main__':
    angle_action_server()