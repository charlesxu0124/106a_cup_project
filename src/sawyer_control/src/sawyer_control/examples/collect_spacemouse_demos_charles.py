#!/usr/bin/python3
# from sawyer_control.envs.sawyer_grip_wrist_env import SawyerGripWristEnv
# from sawyer_control.envs.sawyer_grip_stub import SawyerGripEnv
from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control.envs.sawyer_grip_wrist_env import SawyerGripWristEnv


import numpy as np
import time
# from rlkit.demos.collect_demo import SpaceMouseExpert
from widowx_envs.scripts.spacemouse_teleop import SpaceMouseExpert
import gym
import time
import rospy
import cv2

import sys
import os.path
from os import path
import datetime

# from sawyer_control.envs.sawyer_insertion_refined_USB_sparse_RLonly import SawyerHumanControlEnv


def collect_one_rollout_mdp(env, expert, horizon=200, render=False, pause=0,  threshold=-1,):
    # inp = input('waiting to start rollout')

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
    grasp = False
    while True:
        a, valid, left, right = expert.get_action()
        # print(a)
        accept = left

        if right:
           grasp = not grasp
        # grasp = right  # False
        # if a is not None:
            # grasp = abs(a[5]) > 0.5
        if not valid and not grasp:
            a = np.zeros((6, ))
            valid = True

        if valid:
            traj['observations'].append(o)

            u = np.zeros((5, ))
            u[:3] = a[:3]
            u[3] = -1 if grasp else 1
            u[4] = -a[5]  # rotation
            # print(u)
            # import pdb; pdb.set_trace()
            o, r, done, info = env.step(u)
            traj['actions'].append(u)
            traj['rewards'].append(r)
            traj['next_observations'].append(o)
            traj['terminals'].append(done)
            traj['agent_infos'].append(info)
            traj['env_infos'].append(info)
            ret += r
            t += 1

        if accept or t > horizon:
            break

        if rospy.is_shutdown():
            break

        if render:
            env.render()

        if pause:
            time.sleep(pause)

    return accept, traj


def collect_demos_fixed(env, expert, path='demos.npy', N=10, horizon=200, **kwargs):
    data = []

    i = 0
    while len(data) < N:
        accept, traj = collect_one_rollout_mdp(env, expert, horizon, **kwargs)
        # accept = True
        if accept:
            data.append(traj)
            print('accepted trajectory length', len(traj['observations']))
            # print('last reward', traj['rewards'][-1])
            print('accepted', len(data), 'trajectories')
            # print('total rewards', sum(traj['rewards']))
            # import pdb; pdb.set_trace()
            # break
        else:
            print('discarded trajectory')

    np.save(path, data)
    # pickle.dump(data, open(path, 'wb'), protocol=2)


if __name__ == '__main__':
    filename = sys.argv[1]
    # save_filename = '/media/ashvin/data1/s3doodad/demos/icra2021/dataset_v5_simpledrawers/obj_%s.npy' % filename
    # save_filename = '/media/ashvin/data1/s3doodad/demos/icra2021/dataset_v4b/obj_%s.npy' % filename
    output_dir = '/home/tesla/106a_ws/data/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_filename = os.path.join(
        output_dir,
        'demos_%s_%s.npy' % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), filename)
    )
    print('Saving to: %s ...' % (save_filename))

    if os.path.exists(save_filename):
        print('File already exists at %s.' % (save_filename))
        c = input('Do you want to overwrite?')
        if c in ['y', 'Y']:
            pass
        else:
            exit

    scale = 1000
    expert = SpaceMouseExpert(
        xyz_dims=3,
        xyz_remap=[0, 1, 2],
        xyz_scale=[scale, scale, scale],
        rot_scale=scale,
    )
    env = SawyerGripWristEnv(
        action_mode='position',
        config_name='charles_config',
        reset_free=False,
        position_action_scale=0.01,
        max_speed=0.4,
        # step_sleep_time=0.2,
    )
    # import pdb; pdb.set_trace()
    collect_demos_fixed(env,
                        expert,
                        path=save_filename,
                        N=10,
                        #horizon=100,
                        horizon=200,
                        pause=0.01)

    exit()