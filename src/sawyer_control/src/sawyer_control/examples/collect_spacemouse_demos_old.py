#!/usr/bin/python3
from sawyer_control.envs.sawyer_grip import SawyerGripEnv
# from sawyer_control.envs.sawyer_grip_wrist_env import SawyerGripWristEnv
# from sawyer_control.envs.sawyer_grip_stub import SawyerGripEnv
import numpy as np
import time
# from rlkit.demos.collect_demo import SpaceMouseExpert
from rlkit.demos.spacemouse.input_server import SpaceMouseExpert
import gym
import time
import rospy
import cv2

import sys
import os.path
from os import path

filename = sys.argv[1]
save_filename = "/home/ashvin/data/s3doodad/demos/icra2021/obj_%s.npy" % filename
print("Exists: %s ? %s" % (save_filename, str(path.exists(save_filename))))

# from sawyer_control.envs.sawyer_insertion_refined_USB_sparse_RLonly import SawyerHumanControlEnv

def collect_one_rollout_mdp(env, expert, horizon=200, render=False, pause=0,  threshold=-1,):
    # inp = input("waiting to start rollout")

    o = env.reset()

    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )
    ret = 0
    t = 0
    accept = False
    #grasp = 0
    while True:
        a, valid, right, left = expert.get_action(o)
        accept = left
        reset = False # right

        #if right:
        #    grasp = not grasp
        grasp = right # False
        # if a is not None:
        #     grasp = abs(a[5]) > 0.5
        if not valid and grasp:
            a = np.zeros((6, ))
            valid = True

        if valid:
            traj["observations"].append(o)

            u = np.zeros((5, ))
            u[:3] = a[:3]
            u[3] = -1 if grasp else 1
            u[4] = -a[-1] # rotation
            #print(u)
            o, r, done, info = env.step(u)

            traj["actions"].append(u)
            traj["rewards"].append(r)
            traj["next_observations"].append(o)
            traj["terminals"].append(done)
            traj["agent_infos"].append(info)
            traj["env_infos"].append(info)
            ret += r
            t += 1

        if accept or reset or t > horizon:
            break

        if rospy.is_shutdown():
            break

        if render:
            env.render()

        if pause:
            time.sleep(pause)

    return accept, traj

def collect_demos_fixed(env, expert, path="demos.npy", N=10, horizon=200, **kwargs):
    data = []

    i = 0
    while len(data) < N:
        accept, traj = collect_one_rollout_mdp(env, expert, horizon, **kwargs)
        if accept:
            data.append(traj)
            print("accepted trajectory length", len(traj["observations"]))
            # print("last reward", traj["rewards"][-1])
            print("accepted", len(data), "trajectories")
            # print("total rewards", sum(traj["rewards"]))
        else:
            print("discarded trajectory")

    np.save(path, data)
    # pickle.dump(data, open(path, "wb"), protocol=2)

if __name__ == '__main__':
    scale = 1
    expert = SpaceMouseExpert(
        xyz_dims=3,
        xyz_remap=[0, 1, 2],
        xyz_scale=[scale, scale, scale],
    )
    env = SawyerGripEnv(
        action_mode='position',
        config_name='ashvin_config',
        reset_free=False,
        position_action_scale=0.05,
        max_speed=0.4,
        step_sleep_time=0.2,
    )

    collect_demos_fixed(env, expert, 
        path=save_filename,
        N=5, horizon=200, pause=0.01)

    exit()