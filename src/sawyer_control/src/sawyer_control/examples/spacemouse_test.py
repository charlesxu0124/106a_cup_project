#!/usr/bin/python3
# from sawyer_control.envs.sawyer_grip import SawyerGripEnv
from sawyer_control.envs.sawyer_grip_stub import SawyerGripEnv
import numpy as np
import time

from rlkit.demos.spacemouse.input_server import SpaceMouseExpert
# from rlkit.demos.collect_demo import SpaceMouseExpert
import gym
import time
import rospy
import cv2

# from sawyer_control.envs.sawyer_insertion_refined_USB_sparse_RLonly import SawyerHumanControlEnv

if __name__ == '__main__':
    scale = 0.5
    expert = SpaceMouseExpert(
        xyz_dims=3,
        xyz_remap=[0, 1, 2],
        xyz_scale=[scale, scale, scale],
    )

    # env = SawyerGripEnv(
    #     action_mode='position',
    #     config_name='ashvin_config',
    #     reset_free=False,
    #     position_action_scale=0.01,
    #     max_speed=0.4,
    # )

    # o = env.reset()
    o = None

    for i in range(10000):
        a, valid, reset, grasp = expert.get_action(o)

        if not valid and grasp:
            a = np.zeros((3, ))
            valid = True

        if valid:
            u = np.zeros((4, ))
            u[:3] = a[:3]
            u[3] = -1 if grasp else 1
            print(a, valid, reset, grasp)
            # o, r, done, info = env.step(u)
            time.sleep(0.05)
        else:
            print(a, valid, reset, grasp)

        if reset:
            env.reset()

        if rospy.is_shutdown():
            break

        time.sleep(0.01)

    exit()