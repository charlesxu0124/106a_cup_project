#!/usr/bin/env python
import rospy
import intera_interface as ii
from sawyer_control.srv import angle_action
from sawyer_control.srv import *
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from scipy.interpolate import CubicSpline

# constants for robot control
max_vel_mag = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
max_accel_mag = np.array([3.5, 2.5, 5, 5, 5, 5, 5])

class CSpline:
    def __init__(self, points, duration=1., bc_type='clamped'):
        n_points = points.shape[0]
        self._duration = duration
        self._cs = CubicSpline(np.linspace(0, duration, n_points), points, bc_type=bc_type)

    def get(self, t):
        t = np.array(min(t, self._duration))

        return self._cs(t), self._cs(t, nu=1), self._cs(t, nu=2)


class JointController(object):

    def __init__(self,
                 limb,
                 rate = 1000
                 ):

        # control parameters
        self._rate = rate # Hz

        # create our limb instance
        self._limb = limb

        # initialize parameters
        self.imp_ctrl_publisher = rospy.Publisher('/desired_joint_pos', JointState, queue_size=1)
        self.imp_ctrl_release_spring_pub = rospy.Publisher('/release_spring', Float32, queue_size=10)


    def move_with_impedance(self, des_joint_angles):
        """
        non-blocking
        """
        js = JointState()
        js.name = self._limb.joint_names()
        js.position = [des_joint_angles[n] for n in js.name]
        self.imp_ctrl_publisher.publish(js)


    def move_with_impedance_sec(self, cmd, duration=2.0):
        jointnames = self._limb.joint_names()
        prev_joint = [self._limb.joint_angle(j) for j in jointnames]
        new_joint = np.array([cmd[j] for j in jointnames])
        control_rate = rospy.Rate(self._rate)
        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds

        waypoints = np.array([prev_joint] + [new_joint])
        
        spline = CSpline(waypoints, duration)

        while rospy.get_time() < finish_time:
            int_joints, velocity, acceleration = spline.get(rospy.get_time() - start_time)
            cmd = dict(list(zip(self._limb.joint_names(), list(int_joints))))
            self.move_with_impedance(cmd)
            control_rate.sleep()


def execute_action(action_msg):
    action = action_msg.angles
    joint_names = arm.joint_names()
    joint_to_values = dict(zip(joint_names, action))
    duration = action_msg.duration
    controller.move_with_impedance_sec(joint_to_values, duration=duration)
    return angle_actionResponse(True)

def angle_action_server():
    rospy.init_node('angle_action_server', anonymous=True)
    global arm
    global controller
    arm = ii.Limb('right')
    arm.set_joint_position_speed(0.1) 
    controller = JointController(arm)
    s = rospy.Service('angle_action', angle_action, execute_action)
    rospy.spin()



if __name__ == '__main__':
    angle_action_server()