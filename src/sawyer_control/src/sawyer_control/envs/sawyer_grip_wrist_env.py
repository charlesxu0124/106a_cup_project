from skimage import data, color
from PIL import Image
import torchvision.transforms.functional as F
from skimage.transform import rescale, resize, downscale_local_mean
from collections import OrderedDict
import numpy as np
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from sawyer_control.core.eval_util import get_stat_in_paths, \
    create_stats_ordered_dict
# from sawyer_control.grippers.wsg_gripper import WSG50Gripper
from gym.spaces import Box
import rospy
from gym.spaces import Dict
import threading
from sawyer_control.helper.video_capture import VideoCapture

from pyquaternion import Quaternion
import time
import cv2
from scipy import interpolate
from scipy.optimize import linear_sum_assignment

scripted_actions = [[ 0.00285714,  0.        , -0.        ,  1.        , -0.        ],
       [ 0.04571429,  0.        , -0.        ,  1.        , -0.01142857],
       [-0.10571429,  0.15714286,  1.        ,  1.        ,  0.02      ],
       [ 0.        ,  0.09142857,  1.        ,  1.        ,  0.04285714],
       [-0.09142857,  0.13142857,  1.        ,  1.        ,  0.06571429],
       [-0.1       ,  0.11142857,  1.        ,  1.        ,  0.08      ],
       [ 0.        ,  0.01714286, -0.        ,  1.        , -0.26      ],
       [ 0.00285714,  0.        , -0.00285714,  1.        , -1.        ],
       [ 0.01714286, -0.00857143, -0.        ,  1.        ,  0.00285714],
       [ 0.        ,  0.67142857,  0.29142857,  1.        , -0.        ],
       [ 0.        ,  0.63428571,  0.32857143,  1.        , -0.        ],
       [ 0.        ,  0.61428571,  0.30285714,  1.        , -0.        ],
       [ 0.        ,  0.61428571,  0.27428571,  1.        , -0.        ],
       [ 0.        ,  0.69428571, -0.        ,  1.        , -0.        ],
       [ 0.        ,  1.        , -0.34285714,  1.        , -0.        ],
       [ 0.        ,  1.        , -0.31142857,  1.        , -0.        ],
       [-0.15714286,  1.        , -0.28285714,  1.        , -0.        ],
       [ 0.02285714,  0.        , -0.        ,  1.        , -0.        ],
       [-0.53714286, -0.20857143, -0.11714286,  1.        , -0.15142857],
       [-0.06571429, -0.01428571, -0.        ,  1.        , -0.        ],
       [ 0.        , -0.06285714, -0.        ,  1.        ,  0.00857143],
       [-0.14857143, -0.79714286, -0.        ,  1.        , -0.        ],
       [ 0.        , -0.02571429,  0.00285714,  1.        ,  0.00285714],
       [ 0.        , -0.09142857, -0.06857143,  1.        , -0.        ],
       [-0.29714286,  0.12      , -0.        ,  1.        , -0.        ],
       [-0.16285714,  0.08571429,  0.01142857,  1.        , -0.        ],
       [-0.00857143,  0.02      ,  0.00571429,  1.        , -0.        ],
       [ 0.        ,  0.03714286, -1.        ,  1.        ,  0.00571429],
       [ 0.        ,  0.03142857, -1.        ,  1.        ,  0.02285714],
       [ 0.02285714,  0.        , -0.43428571,  1.        ,  0.05428571],
       [ 0.        ,  0.        , -0.05428571,  1.        ,  0.04285714],
       [ 0.06285714,  0.        , -0.76857143,  1.        ,  0.07714286],
       [ 0.        ,  0.        , -0.08571429,  1.        , -0.        ],
       [ 0.        ,  0.        , -0.00857143,  1.        ,  0.00285714],
       [ 0.16285714, -0.02      ,  0.02857143,  1.        ,  0.03714286],
       [ 0.78571429,  0.        ,  0.11142857,  1.        , -0.        ],
       [ 0.79142857,  0.01428571,  0.09142857,  1.        , -0.        ],
       [ 0.81714286,  0.00285714,  0.03428571,  1.        , -0.        ],
       [ 0.73428571,  0.04285714,  0.00857143,  1.        , -0.        ],
       [ 0.48571429,  0.        , -0.        ,  1.        , -0.        ]]
scripted_actions = np.array(scripted_actions)
TAPE_GOAL = np.array([(314, 347), (282, 335), (282, 306), (305, 291), (336, 289)])
SPLINE_GOAL =  np.array([[312.0, 339.0], [309.29081707037676, 338.9515131318601], 
[302.5402286880636, 338.5741153437611], [293.81367546289545, 337.5199008601837], 
[285.17659800470705, 335.4409639056089], [278.6870639201883, 331.99171057838504], 
[275.51100743548625, 327.10628371482727], [275.08170703767735, 321.26211651011187], 
[276.63337012892345, 314.9990627538375], [279.4002041113865, 308.8569762356029], 
[282.64149292899845, 303.3640472372066], [286.1932809863995, 298.780205361048], 
[290.46837314893884, 295.0971195301273], [295.9046508237352, 292.29479515964425], 
[302.93999541790765, 290.352376647991], [311.86495324183034, 289.2372690729594], 
[321.6877095786557, 288.77955970258057], [330.7561702039031, 288.74473580072066], 
[337.412840376565, 288.89686126674025], [340.00000000000006, 289.0]])

UL = (180, 161)
UR = (561, 164)
LL = (140, 438)
LR = (593, 466)

class CSpline:
    def __init__(self, points, skip=[], duration=1., bc_type='clamped'):
        n_points = points.shape[0] + len(skip)
        self._duration = duration
        while True:
            if skip == [] or skip[-1] != n_points - 1:
                break
            else:
                skip = skip[:-1]
                n_points -= 1
        x = np.linspace(0, duration, n_points)
        x = np.delete(x, skip)
            
        # self._cs = interpolate.CubicSpline(np.linspace(0, duration, n_points-len(skip)), points, bc_type=bc_type)
        self._cs = interpolate.CubicSpline(x, points, bc_type=bc_type)

    def get_cs(self):
        return self._cs

    def get(self, t):
        t = np.array(min(t, self._duration))

        return self._cs(t), self._cs(t, nu=1), self._cs(t, nu=2)


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
                 reset_every=1,
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
        self.indicator_threshold = indicator_threshold
        self.reward_type = reward_type
        self._state_goal = np.array(fixed_goal)
        self.reset_angles = self.config.RESET_ANGLES
        self.gripper_action = 1
        # TODO(kuanfang): Set this to 0 when the gripper is unused.
        USE_GRIPPER = False
        if USE_GRIPPER:
            print("initializing gripper")
            self.gripper = WSG50Gripper()
        else:
            print("NOT initializing gripper")
            self.gripper = None

        print("Done initializing gripper")
        resource = "/dev/video0"
        # self.global_rotation = 0
        # self.global_rotation_bound = [-1.5, 4] # figure out these values
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
        self.action_space = Box(-np.ones((5, )), np.ones((5, )))
        self.step_sleep_time = step_sleep_time
        self.crop = get_crop_fn(crop_version_str)
        self.max_rotation = max_rotation
        self.traj_counter = 1

        # goalbisim
        self.max_episode_steps = 200
        self.obs_img_dim = 64
        self.reset_every = reset_every
        print('reset_every: ', reset_every)
        # input()  # TODO(kuanfang): This is necessary?

    def sample_goals(self, batch_size):
        return {'state_desired_goal': np.zeros((3,))}

    def reset(self, seed=None):
        self._reset_robot()
        self._num_steps_this_episode = 0
        self._state_goal = np.zeros((3,))  # self.sample_goal()
        return self._get_obs()

    def step(self, action):
        gripper_action = action[3]
        if self.gripper is not None:
            self.gripper.do_cmd(gripper_action)
        self.gripper_action = gripper_action

        self._act(action)
        time.sleep(self.step_sleep_time)
        observation = self._get_obs()
        reward = self.compute_rewards(observation['observation'])
        print("REWARD: ", reward)
        # reward = 0
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def _act(self, action, clip=True, orientation=None):

       # if self._num_steps_this_episode < len(scripted_actions):
            # action = scripted_actions[self._num_steps_this_episode]

        # for ind in range(3):
        #     if np.abs(action[ind]) < 0.5:
        #         action[ind] = 0
        #     else:
        #         if action[ind] > 0:
        #             action[ind] = 1.0
        #         else:
        #             action[ind] = -1.0

        # print('action: ', action)
        # print(self.action_mode)

        if self.action_mode == 'position':
            # self.pos_control_ee_orientation)
            self._position_act(action, clip, )
        else:
            self._torque_act(action * self.torque_action_scale)
        self._num_steps_this_episode += 1
        return

    def _position_act(self, action, clip=True, orientation=None):
        joint_angles, _, ee_pose, _, _ = self.request_observation()
        self.ee_pos = ee_pose

        dpos = action[:3] * self.position_action_scale
        rotation = action[4]
        dr = rotation * self.max_rotation

        print('step: %d, rotation: %r, dr: %r' 
              % (self._num_steps_this_episode, rotation, dr))  # TODO

        endeffector_pos = ee_pose[:3]
        ee_quat = Quaternion(ee_pose[3:])
        target_ee_quat = Quaternion(axis=[1, 0, 0], angle=dr) * ee_quat
        target_ee_pos = (endeffector_pos + dpos)

        ## Restrict gripper rotation within pi/2
        min_value = -np.pi
        max_value = np.pi
        ee_euler = ee_quat.angle
        target_ee_euler = target_ee_quat.angle
        old_target_ee_euler = target_ee_euler
        target_ee_euler = target_ee_euler % (2 * np.pi)
        offset = old_target_ee_euler - target_ee_euler
        target_ee_euler = np.clip(target_ee_euler, min_value, max_value)
        target_ee_euler = target_ee_euler + offset
        target_ee_quat = Quaternion(axis=[1, 0, 0], angle=target_ee_euler)
        target_ee_euler = target_ee_quat.angle

        orientation = target_ee_quat.elements

        # print('input ee target pos: ', target_ee_pos)
        # import pdb; pdb.set_trace()
        if clip:
            target_ee_pos = np.clip(
                target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
        if orientation is not None:
            target_ee_pos = np.concatenate((target_ee_pos, orientation))
        else:
            target_ee_pos = np.concatenate(
                (target_ee_pos, self.config.QUATERNIONS_GOAL))
        angles = self.request_ik_angles(target_ee_pos, joint_angles)
        # print(angles)
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
                noise = np.clip(np.random.normal(0, 0.005, 7), -0.001, 0.001)
                angles += noise
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

        if self.traj_counter % self.reset_every == 0:
            inp = input("reset the scene, enter when it is done")
            while inp == "r":
                print("reset again")
                for _ in range(10):
                    self.step(np.array([0, 0, 0.5, 0, 0]))
                self.request_reset_angle_action()
                inp = input("reset the scene, enter when it is done")
        else:
            inp = input("start next traj")
            while inp == "r":
                print("reset again")
                for _ in range(10):
                    self.step(np.array([0, 0, 0.5, 0, 0]))
                self.request_reset_angle_action()
                inp = input("reset the scene, enter when it is done")

        self.traj_counter += 1

    def _set_action_space(self):
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
            dtype=np.float32,
        )

    def _get_obs(self):
        joint_angles, _, endeff, _, _ = self.request_observation()
        angles = np.concatenate((joint_angles, [self.gripper_action,]))
        image = self.cap.read()  # (480, 640, 3) # let renderer generate image
        frame = np.copy(image)
        tape, spline = self.find_rope(frame)
        obs = dict(joint_angles=angles, endeff=endeff, tape=tape, spline=spline)
        # cv2.imshow("test",image)
        # key = cv2.waitKey(1)
        obs_dict = dict(
            observation=obs,
            desired_goal=obs,
            achieved_goal=obs,
            state_observation=obs,
            state_desired_goal=obs,
            state_achieved_goal=obs,
            image_observation=image,  # [::10, 90:570:10, :].flatten(),
            hires_image_observation=image,  # [::10, 90:570:10, :].flatten(),
        )
        return obs_dict

    def get_observation(self):
        return self._get_obs()

    def render_obs(self):
        return self.get_image()

    def get_image(self, width=84, height=84):
        image = self.cap.read()
        return self.crop(image)


    def compute_rewards(self, obs):

        # tape = obs['tape']
        # spline = obs['spline']
        # skip = [i for i in range(len(tape)) if tape[i][0] == None] 
        # tape = np.delete(tape, skip, axis=0)
        # goal = np.delete(TAPE_GOAL, skip, axis=0)
        # return -np.linalg.norm(tape - goal)
        reward = 0
        goal = np.array(TAPE_GOAL[1:4], dtype=np.float)
        tape = np.array([_ for _ in obs['tape'][1:4] if all(_)], dtype=np.float)
        # print(tape)
        # print(goal)
        if len(tape) > 0:
            cost0 = np.linalg.norm(tape-goal[0], axis=1)
            cost1 = np.linalg.norm(tape-goal[1], axis=1)
            cost2 = np.linalg.norm(tape-goal[2], axis=1)
            cost = np.vstack((cost0, cost1, cost2))
            row_ind, col_ind = linear_sum_assignment(cost)
            reward -= cost[row_ind, col_ind].sum()
        # import pdb; pdb.set_trace()
        if all(obs['tape'][0]):
            reward -= np.linalg.norm(obs['tape'][0]-TAPE_GOAL[0], axis=0)
        if all(obs['tape'][4]):
            reward -= np.linalg.norm(obs['tape'][4]-TAPE_GOAL[4], axis=0)
        return reward

    def find_rope(self, frame):
        # cap, q = self.streams[self.name]
        # ret, frame = cap.read()
        sagments = []
        #seperate color blocks
        blurred = cv2.GaussianBlur(frame, (9, 9), sigmaX=10, sigmaY=10)
        # cv2.imshow("blurred", blurred)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv", hsv)

        #blue
        lower = (100 , 150, 75)
        upper = (130, 255, 255)
        blue = self.find_color("blue", hsv, lower, upper, sagments, frame)
        sagments.append(blue)

        #green
        lower = (45, 20, 10)
        upper = (90, 255, 255)
        greens = self.find_color("green", hsv, lower, upper, sagments, frame)
        for green in greens:
            sagments.append(green)

        #orange
        lower = (150, 10, 50)
        upper = (15, 255, 255)
        # # yellow
        # lower = (10, 0, 0)
        # upper = (40, 255, 255)
        orange = self.find_color("orange", hsv, lower, upper, sagments, frame)
        sagments.append(orange)

        for sagment in sagments:
            if all(sagment):
                cv2.circle(frame, sagment, 2, (0,0,255), thickness = 3)
    

        x = [i[0] for i in sagments if i[0] != None]
        y = [i[1] for i in sagments if i[1] != None]
        skip = [i for i in range(len(sagments)) if sagments[i][0] == None] 
        
        
        
        # x.insert(0, 393)
        # y.insert(0, 338)
        # tck = interpolate.splprep([x, y], u=0)
        # if len(x) > 2:
        #     points = np.c_[x, y]
        #     spline = CSpline(np.asarray(points), skip=skip)
        #     cs = spline.get_cs()
        #     xs = np.linspace(0, 1, 20)
        #     points = np.ndarray.tolist(cs(xs))
        #     for point in points:
        #         cv2.circle(frame, (int(point[0]), int(point[1])), 1, (255, 0, 0), thickness=2)

        cv2.imshow("result", frame)
        key = cv2.waitKey(1)
        # print(sagments)
        return np.array(sagments), []

    def find_color(self, name, hsv, lower, upper, sagments, frame):
        # cv2.imshow("hsv", hsv)
        # frame = self._get_obs()['image_observation']
        if lower[0] > upper[0]:
            mask1 = cv2.inRange(hsv, (0, lower[1], lower[2]), upper)
            mask2 = cv2.inRange(hsv, lower, (179, upper[1], upper[2]))
            res1 = cv2.bitwise_and(frame, frame, mask=mask1)
            res2 = cv2.bitwise_and(frame, frame, mask=mask2)
            res = cv2.bitwise_or(res1, res2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, lower, upper)
            res = cv2.bitwise_and(frame, frame, mask= mask)
        # cv2.imshow("blue", hsv)
        cv2.imshow(name, res)
        contours, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        backups = []
        if name == "green":
            greens = []
            for cnt in contours:
                cnt = np.asarray(cnt)
                center = np.mean(cnt,0)[0]
                if not self.inRegion(center):
                    pass
                elif len(cnt) < 3 or len(cnt) > 40:
                    pass
                elif len(greens) > 0 and min([np.linalg.norm(center - prev_green) for prev_green in greens]) < 10:
                    backups.append(tuple([int(c) for c in center]))
                elif len(greens) < 3:
                    greens.append(tuple([int(c) for c in center]))
                else:
                    break
            if len(greens) < 3:
                greens = greens + backups + [(None, None), (None, None), (None, None)]
            return greens[:3]
        else:
            result = (0, 0)
            backup = (0, 0)
            for cnt in contours[::-1]:
                cnt = np.asarray(cnt)
                center = np.mean(cnt,0)[0]
                if not self.inRegion(center):
                    pass
                elif len(cnt) < 4 or len(cnt) > 40:
                    pass
                elif np.count_nonzero(sagments) > 0 and min(
                    [np.linalg.norm(center - prev_centers) for prev_centers in sagments+[(272, 307),(293, 308)] if prev_centers[0] != None]) < 10:
                    backup = tuple([int(c) for c in center])
                elif center[1] > result[1]:
                    result = tuple([int(c) for c in center])
            if all(result) == 0 and all(backup) == 0:
                return (None, None)
            elif all(result) == 0:
                return backup
            return result


    def inRegion(self, pos):
        cof = np.polyfit([UL[0], LL[0]], [UL[1], LL[1]], 1)
        if pos[1] < cof[0]*pos[0] + cof[1]:
            return False
        cof = np.polyfit([UL[0], UR[0]], [UL[1], UR[1]], 1)
        if pos[1] < cof[0]*pos[0] + cof[1]:
            return False
        cof = np.polyfit([LL[0], LR[0]], [LL[1], LR[1]], 1)
        if pos[1] > cof[0]*pos[0] + cof[1]:
            return False
        cof = np.polyfit([UR[0], LR[0]], [UR[1], LR[1]], 1)
        if pos[1] < cof[0]*pos[0] + cof[1]:
            return False
        return True

    def _set_observation_space(self):
        if self.action_mode == 'position':
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
            self._position_act(action * self.position_action_scale,
                               clip, self.pos_control_ee_orientation)
            time.sleep(0.05)

        tmp = "r"
        while tmp == "r":
            tmp = input("Press Enter When Ready")

        return self._get_endeffector_pose()[:3]

    # choose larger t here
    def reach_goal_with_tol(self, goal, tol=0.001, t=10, orientation=None):
        self._state_goal = goal
        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + t  # in seconds
        time = rospy.get_time()
        dist = np.inf
        while dist > tol and time < finish_time:
            err = goal - self._get_endeffector_pose()[:3]
            dist = np.linalg.norm(err)
            print('error [m]: ', dist)
            self._position_act(err, clip=False, orientation=orientation)
            time = rospy.get_time()

    def get_contextual_diagnostics(self, a, b):
        return {}

    def demo_reset(self, seed=None):
        pass


def get_crop_fn(version_str):
    ## VAL CROP
    def crop_val_original(img):
        img = resize(img[0:270, 90:570, ::-1], (48, 48), anti_aliasing=True) * 255
        img = img.astype(np.uint8)
        return img
        # return img.transpose([2, 1, 0]).flatten()

    def crop_goalbisim(img):
        img = Image.fromarray(img[0:270, 90:570, ::-1], mode='RGB')
        img = F.resize(img, (64, 64), Image.ANTIALIAS)
        img = np.array(img)
        #img = img.transpose([2, 1, 0]) #.flatten()
        return img

    ## VAL "in-distribution" crop
    def crop_val_torch(img):
        #img *= 255
        img = Image.fromarray(img[0:270, 90:570, ::-1], mode='RGB')
        img = F.resize(img, (48, 48), Image.ANTIALIAS)
        img = np.array(img)
        #img = img.transpose([2, 1, 0]) #.flatten()
        return img

    # def crop(img):
    #     img = resize(img[0:270, 90:570, ::-1], (48, 48), anti_aliasing=True) * 255
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
        crop_goalbisim=crop_goalbisim,
    )[version_str]
