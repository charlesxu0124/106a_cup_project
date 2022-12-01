#!/usr/bin/python3
from sawyer_control.envs.sawyer_grip_wrist_tilt_env import SawyerGripWristEnv
import numpy as np
import time

env = SawyerGripWristEnv(
    action_mode='position',
    config_name='ashvin_config',
    reset_free=False,
    position_action_scale=0.01,
    max_speed=0.4,
)
# import ipdb; ipdb.set_trace()
env.reset()

N = 20
T = 0.2
X = 0.5
Y = 0.5
Z = 1
for i in range(N):
    env.step(np.array([X, 0, 0, 0, 0, -1]))
    # print(i)
    time.sleep(T)
for i in range(N):
    env.step(np.array([-X, 0, 0, 0, 0, 1]))
    # print(i)
    time.sleep(T)
for i in range(N):
    env.step(np.array([0, Y, 0, 0, 0, 1]))
    # print(i)
    time.sleep(T)
for i in range(N):
    env.step(np.array([0, -Y, 0, 0, 0, -1]))
    # print(i)
    time.sleep(T)
for i in range(N):
    env.step(np.array([0, 0, Z, 0, 0, 0]))
    # print(i)
    time.sleep(T)
for i in range(N):
    env.step(np.array([0, 0, -Z, 0, 0, 0]))
    # print(i)
    time.sleep(T)