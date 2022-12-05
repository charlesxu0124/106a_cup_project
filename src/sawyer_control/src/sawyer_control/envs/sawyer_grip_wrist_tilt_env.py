from collections import OrderedDict
import numpy as np
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from sawyer_control.core.eval_util import get_stat_in_paths, \
    create_stats_ordered_dict
# from sawyer_control.grippers.wsg_gripper import WSG50Gripper
from gym.spaces import Box
import rospy
# from multiworld.core.multitask_env import MultitaskEnv
from gym.spaces import Dict
import threading
from sawyer_control.helper.video_capture import VideoCapture

from pyquaternion import Quaternion
import time

class SawyerGripWristEnv(SawyerEnvBase, ):
    def __init__(self,
                 fixed_goal=(1, 1, 1),
                 step_sleep_time=0.05,
                 indicator_threshold=.05,
                 reward_type='hand_distance',
                 goal_low=None,
                 goal_high=None,
                 crop_version_str="crop_val_original",
                 max_rotation=0.2,
                 max_tilt=0.2,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, **kwargs)
        if self.action_mode == 'torque':
            if goal_low is None:
                goal_low = self.config.TORQUE_SAFETY_BOX.low
            if goal_high is None:
                goal_high = self.config.TORQUE_SAFETY_BOX.high
        else:
            if goal_low is None:
                goal_low = self.config.POSITION_SAFETY_BOX.low
            if goal_high is None:
                goal_high = self.config.POSITION_SAFETY_BOX.high
        self.goal_space = Box(goal_low, goal_high, dtype=np.float32)
        imsize = 480 * 640 * 3
        self.image_space = Box(np.zeros((imsize, )), np.ones((imsize, )))
        self.indicator_threshold=indicator_threshold
        self.reward_type = reward_type
        self._state_goal = np.array(fixed_goal)
        self.reset_angles = self.config.RESET_ANGLES
        self.gripper_action = 1
        USE_GRIPPER = False
        if USE_GRIPPER:
            print("initializing gripper")
            self.gripper = WSG50Gripper()
        else:
            print("NOT initializing gripper")
            self.gripper = None
        resource = "/dev/video0"
        self.cap = VideoCapture(resource)
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.observation_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
            ('image_observation', self.observation_space),
        ])
        self.action_space = Box(-np.ones((6, )), np.ones((6, )))
        self.step_sleep_time = step_sleep_time
        self.crop = get_crop_fn(crop_version_str)
        self.max_rotation = max_rotation
        self.max_tilt = max_tilt

    def step(self, action):
        gripper_action = action[3]
        if self.gripper is not None:
            self.gripper.do_cmd(gripper_action)
        self.gripper_action = gripper_action

        self._act(action)
        time.sleep(self.step_sleep_time)
        observation = self._get_obs()
        # reward = self.compute_reward(action, self.convert_ob_to_goal(observation), self._state_goal)
        reward = 0
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def _act(self, action, clip = True, orientation = None):
        if self.action_mode == 'position':
            self._position_act(action, clip, ) # self.pos_control_ee_orientation)
        else:
            self._torque_act(action * self.torque_action_scale)
        return

    def _position_act(self, action, clip = True, orientation = None):
        joint_angles, _, ee_pose, _, _ = self.request_observation()
        self.ee_pos = ee_pose

        dpos = action[:3] * self.position_action_scale
        rotation = action[4]
        tilt = action[5]
        dr = rotation * self.max_rotation
        dt = tilt * self.max_tilt
        print(rotation, dr, dt)
        endeffector_pos = ee_pose[:3]
        ee_quat = Quaternion(ee_pose[3:])
        target_ee_quat = Quaternion(axis=[0, 0, 1], angle=dt) * ee_quat
        target_ee_quat = Quaternion(axis=[1, 0, 0], angle=dr) * target_ee_quat
        target_ee_pos = (endeffector_pos + dpos)

        orientation = target_ee_quat.elements

        if clip:
            target_ee_pos = np.clip(target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
        if orientation is not None:
            target_ee_pos = np.concatenate((target_ee_pos, orientation))
        else:
            target_ee_pos = np.concatenate((target_ee_pos, self.config.QUATERNIONS_GOAL))
        angles = self.request_ik_angles(target_ee_pos, joint_angles)
        if angles != np.array([]):
            angles = np.concatenate((angles, [self.gripper_action,])) # overloaded with gripper action
            self.send_angle_action(angles, target_ee_pos)

    def _reset_robot(self):
        if self.gripper is not None:
            self.gripper.do_open_cmd()
        if not self.reset_free:
            if self.action_mode == "position":
                current_angle, _, _, _, _ = self.request_observation()
                angles = self.reset_angles
                pos = self.pos_control_reset_position
                target_ee_pos = np.concatenate((pos, self.config.QUATERNIONS_GOAL))
                while np.linalg.norm(current_angle-angles) > 0.01:
                    self.send_angle_action(angles, np.array(target_ee_pos))
                    current_angle, _, ee_pos, _, _ = self.request_observation()

                # self.request_reset_angle_action()
            else:
                self.in_reset = True
                self._safe_move_to_neutral()
                self.in_reset = False

            inp = input("reset done, press enter to continue: ")
            while inp == "r":
                print("reset again")
                for _ in range(10):
                    self.step(np.array([0, 0, 0.5, 0, 0, 0]))
                self.request_reset_angle_action()
                inp = input("reset the scene, enter when it is done")

    def _set_action_space(self):
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
            dtype=np.float32,
        )

    def _get_obs(self):
        endeff = np.zeros(6)
        ee_pos = self._get_endeffector_pose()
        endeff[:3] = ee_pos[:3]
        yaw, pitch, roll = Quaternion(ee_pos[3:]).yaw_pitch_roll
        endeff[3:] = np.array([yaw, pitch, roll])
        image = self.cap.read() # (480, 640, 3) # let renderer generate image
        obs_dict = dict(
            observation=endeff,
            desired_goal=endeff,
            achieved_goal=endeff,
            state_observation=endeff,
            state_desired_goal=endeff,
            state_achieved_goal=endeff,
            image_observation=image, # [::10, 90:570:10, :].flatten(),
            hires_image_observation=image, # [::10, 90:570:10, :].flatten(),
        )
        return obs_dict

    def get_image(self, width=84, height=84):
        image = self.cap.read()
        return self.crop(image)

    def compute_rewards(self, actions, obs, goals):
        distances = np.linalg.norm(obs - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def _get_info(self):
        hand_distance = np.linalg.norm(self._state_goal - self._get_endeffector_pose()[:3])
        return dict(
            hand_distance=hand_distance,
            hand_success=(hand_distance<self.indicator_threshold).astype(float)
        )
    def _set_observation_space(self):
        if self.action_mode=='position':
            lows = np.hstack((
                self.config.END_EFFECTOR_VALUE_LOW['position'],
            ))
            highs = np.hstack((
                self.config.END_EFFECTOR_VALUE_HIGH['position'],
            ))
        else:
            lows = np.hstack((
                self.config.JOINT_VALUE_LOW['position'],
                self.config.JOINT_VALUE_LOW['velocity'],
                self.config.END_EFFECTOR_VALUE_LOW['position'],
                self.config.END_EFFECTOR_VALUE_LOW['angle'],
            ))
            highs = np.hstack((
                self.config.JOINT_VALUE_HIGH['position'],
                self.config.JOINT_VALUE_HIGH['velocity'],
                self.config.END_EFFECTOR_VALUE_HIGH['position'],
                self.config.END_EFFECTOR_VALUE_HIGH['angle'],
            ))

        self.observation_space = Box(
            lows,
            highs,
            dtype=np.float32,
        )

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in []:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    """
    Multitask functions
    """

    @property
    def goal_dim(self):
        return 3

    def convert_obs_to_goals(self, obs):
        return obs[:, -3:]

    def set_to_goal(self, goal):
        print("setting to goal", goal)
        for _ in range(50):
            action = goal - self._get_endeffector_pose()[:3]
            clip = True
            #print(action)
            self._position_act(action * self.position_action_scale, clip, self.pos_control_ee_orientation)
            time.sleep(0.05)

        tmp = "r"
        while tmp == "r":
            tmp = input("Press Enter When Ready")

        return self._get_endeffector_pose()[:3]

    def reach_goal_with_tol(self, goal, tol = 0.001, t = 10, orientation = None): # choose larger t here
        self._state_goal = goal
        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + t  # in seconds
        time = rospy.get_time()
        dist = np.inf
        while dist > tol and time < finish_time:
            err = goal - self._get_endeffector_pose()[:3]
            dist = np.linalg.norm(err)
            print('error [m]: ', dist)
            self._position_act(err, clip = False, orientation = orientation) 
            time = rospy.get_time()

    def get_contextual_diagnostics(self, a, b):
        return {}

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import torchvision.transforms.functional as F
from PIL import Image

def get_crop_fn(version_str):
    ## VAL CROP
    def crop_val_original(img):
        img = resize(img[0:270, 90:570, ::-1], (48, 48), anti_aliasing=True) * 255
        img = img.astype(np.uint8)
        return img
        # return img.transpose([2, 1, 0]).flatten()

    ## VAL "in-distribution" crop
    def crop_val_torch(img):
        #img *= 255
        img = Image.fromarray(img[0:270, 90:570, ::-1], mode='RGB')
        img = F.resize(img, (48, 48), Image.ANTIALIAS)
        img = np.array(img)
        #img = img.transpose([2, 1, 0]) #.flatten()
        return img


    # def crop(img):
    #     img = resize(img[0:270, ::-1], (48, 48), anti_aliasing=True) * 255
    #     img = img.astype(np.uint8)
    #     return img
        # return img.transpose([2, 1, 0]).flatten()

    ## DISCO CROP ## OUT OF DATE. NOW UNUSED
    # def crop(img):
    #     import cv2
    #     img = cv2.resize(img, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
    #     Cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = Cimg # / 256 # .transpose(2,1,0) / 256 # .reshape(1, -1)
    #     return img
    # def crop(img):
    #     img = resize(img[:, :, ::-1], (48, 48), anti_aliasing=True) * 255
    #     img = img.astype(np.uint8)
    #     return img
    #     # return img.transpose([2, 1, 0]).flatten()
    def crop_disco(img):
        img = resize(img[0:270, 90:570, ::-1], (48, 48), anti_aliasing=True) * 255
        img = img.astype(np.uint8)
        return img
        return img.transpose([2, 1, 0]).flatten()
    return dict(
        crop_val_original=crop_val_original,
        crop_val_torch=crop_val_torch,
        crop_disco=crop_disco,
    )[version_str]
