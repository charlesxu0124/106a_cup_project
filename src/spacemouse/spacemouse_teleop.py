"""Uses a spacemouse as action input into the environment.

To use this, first clone robosuite (git@github.com:anair13/robosuite.git),
add it to the python path, and ensure you can run the following file (and
see input values from the spacemouse):

robosuite/devices/spacemouse.py

You will likely have to `pip install hidapi` and Spacemouse drivers.
"""
from widowx_env.widowx.widowx_grasp_env import GraspWidowXEnv
from widowx_env.policies.spacemouse import SpaceMouse
from widowx_env.widowx.env_wrappers import NormalizedBoxEnv

import numpy as np
import os
import argparse
import datetime
from PIL import Image
import pickle

class SpaceMouseExpert():
    def __init__(self, xyz_dims=3, xyz_remap=[0, 1, 2], xyz_scale=[1, 1, 1]):
        """TODO: fill in other params"""

        self.xyz_dims = xyz_dims
        self.xyz_remap = np.array(xyz_remap)
        self.xyz_scale = np.array(xyz_scale)
        self.device = SpaceMouse()
        self.grasp_input = 0.
        self.grasp_output = 1.

    def get_action(self, obs):
        """Must return (action, valid, reset, accept)"""
        state = self.device.get_controller_state()
        dpos, rotation, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["grasp"],
            state["reset"],
        )
        # detect button press
        if grasp and not self.grasp_input:
            # open/close gripper
            self.grasp_output = 1. if self.grasp_output <= 0. else -1.
        self.grasp_input = grasp

        xyz = dpos[self.xyz_remap] * self.xyz_scale

        a = xyz[:self.xyz_dims]
        a = np.concatenate([a, [0., self.grasp_output]])

        valid = not np.all(np.isclose(a, 0))

        return (a, valid, reset, grasp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--object")
    parser.add_argument("-d", "--data-save-directory", type=str,
                        default="~/trainingdata")
    parser.add_argument("--num-trajectories", type=int, default=50000)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--no-save", action='store_true')

    args = parser.parse_args()

    if not args.no_save:
        args.data_save_directory = os.path.expanduser(args.data_save_directory)
        time_string = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        time_string += '_{}'.format(args.object)

        directory_name = time_string
        save_dir = os.path.join(args.data_save_directory, directory_name)
        os.makedirs(save_dir)
        assert args.object, "object argument required for the save directory name"

    env = NormalizedBoxEnv(GraspWidowXEnv(
        {'transpose_image_to_chw': True,
         'wait_time': 0.2,
         'move_duration': 0.2,
         'action_mode': '3trans1rot',
         'return_full_image': True,
         'continuous_gripper': False}
    ))

    spacemouse_policy = SpaceMouseExpert(xyz_scale=[200, -200, 200], xyz_remap=[1, 0, 2])

    o = env.reset()

    num_success = 0
    num_fail = 0
    for i in range(args.num_trajectories):
        print('traj #{}'.format(i))
        obs = env.reset()

        if not args.no_save:
            time_string = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
            current_save_dir = os.path.join(save_dir, time_string)
            os.makedirs(current_save_dir)

            filename = 'episode_{}.pkl'.format(time_string)
            filepath = os.path.join(current_save_dir, filename)

        full_image = obs['full_image']
        obs.pop('full_image')

        observations = []
        actions = []
        rewards = []
        terminals = []
        env_infos = []
        next_observations = []

        done = False
        prev_a = False
        prev_b = False
        j = 0
        while not done:
            action, valid, reset, grasp = spacemouse_policy.get_action()
            action = np.random.normal(loc=action, scale=args.action_noise)
            action = np.clip(action, -1.0, 1.0)
            print(action)
            next_obs, reward, _, info = env.step(action)

            full_image_next = next_obs['full_image']
            next_obs.pop('full_image')

            if not args.no_save:
                im = Image.fromarray(full_image)
                imfilepath = os.path.join(current_save_dir, '{}.jpeg'.format(j+1))
                j += 1
                im.save(imfilepath)

            observations.append(obs)
            next_observations.append(next_obs)
            rewards.append(reward)
            terminals.append(done)
            actions.append(action)
            env_infos.append(info)

            obs = next_obs
            full_image = full_image_next

        if not args.no_save:
            with open(filepath, 'wb') as handle:
                path = dict(
                    observations=observations,
                    actions=actions,
                    rewards=np.array(rewards).reshape(-1, 1),
                    next_observations=next_observations,
                    terminals=np.array(terminals).reshape(-1, 1),
                    env_infos=env_infos,
                )
                pickle.dump(path, handle, protocol=pickle.HIGHEST_PROTOCOL)
