import rospy
import numpy as np
from intera_core_msgs.msg import JointCommand
import intera_interface
from intera_interface import CHECK_VERSION
from scipy.interpolate import CubicSpline
from sawyer_control.srv import observation


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

class PositionController():
    def __init__(self, control_rate):
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        print("Robot enabled...")
        self.limb = intera_interface.Limb("right")
        print("Done initializing controller.")
        self._cmd_publisher = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, queue_size=100)
        self.control_rate = rospy.Rate(control_rate)
        rospy.wait_for_service('observations')

    def move_along_spline_in_vel_mode(self, waypoints, duration=1.5):
        """
        Moves from curent position to final position while hitting waypoints
        :param waypoints: List of arrays containing waypoint joint angles
        :param duration: trajectory duration
        """
        
        jointnames = self.limb.joint_names()
        prev_joint = np.array([self.limb.joint_angle(j) for j in jointnames])
        waypoints = np.array([prev_joint] + waypoints)

        spline = CSpline(waypoints, duration)
        forces = np.zeros(6)

        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds

        time = rospy.get_time()
        counter = 0
        obs_interval = 50
        threshold = 10 # [N], adjust force threshold based on need
        
        while time < finish_time:
            
            # SAFETY FEATURE BY GERRIT SCHOETTLER
            if (counter % obs_interval) == 10:
                try:
                    request = rospy.ServiceProxy('observations', observation, persistent=True)
                    obs = request()
                    forces = np.array(obs.endpoint_wrench)
                except rospy.ServiceException as e:
                    print(e)

            if -threshold <= max(abs(forces)) <= threshold:
                pos, velocity, acceleration = spline.get(time - start_time)
                command = JointCommand()
                command.mode = JointCommand.VELOCITY_MODE   # important difference to previous controller [Gerrit]
                #command.mode = JointCommand.POSITION_MODE  # position mode is less accurate than velocity mode
                command.names = jointnames
                command.position = pos
                command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
                command.acceleration = np.clip(acceleration, -max_accel_mag, max_accel_mag)
                self._cmd_publisher.publish(command)

                self.control_rate.sleep() # sleeping in order to sync to 1000 Hz
                time = rospy.get_time()
            else:
                break

            counter += 1