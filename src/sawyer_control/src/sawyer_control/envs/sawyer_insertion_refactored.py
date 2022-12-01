from collections import OrderedDict
import numpy as np
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from sawyer_control.core.eval_util import get_stat_in_paths, \
    create_stats_ordered_dict
from sawyer_control.core.multitask_env import MultitaskEnv
from sawyer_control.configs.config import config_dict as config
from gym.spaces import Box
import rospy
from timeit import default_timer as timer

#for usb detection
import pyudev
import threading
import logging as LOGGER
from std_msgs.msg import Int8
from sawyer_control.helper.usb_detection import USBDetector

from rlkit.samplers.util import rollout
import joblib
from rlkit.envs.wrappers import NormalizedBoxEnv
from PIL import Image

class SawyerHumanControlEnv(SawyerEnvBase, MultitaskEnv):
    def __init__(self,
                observation_type = 'images',   # 'images', 'position_xy', 'position_xyz', 'position_xyzf'
                action_type = 'position_xyz',  # 'position_xy', 'position_xyz'
                reward_type='goal_image',      # 'goal_image', 'hand_distance', 'hand_distance_z', 'hand_distance_shaped', usb_sparse', 'sparse_regularized', 'sparse'
                config_name = 'calibration_config',

                max_stepsize=[0.001, 0.001, 0.001], # scaling factors for xyz movement, actions are in [-1,1] 
                human_ctrl_gain=[1.0, 1.0, 1.0],    # gain of the human controller

                RLoff = True,                       # ignore RL controller 
                Human_off = False,                  # ignore Human controller 
                lift_human_controller_goal = False, # move human controller goal to top of object 
                human_z_off = False,                # turn z-control of human controller off when RL kicks in 
                finish_when_inserted = False,       # finish insertion when indication height is reached  
                approach_target = True,             # approach target after reset

                random_reset = False,               # adds noise to the first step          
                noisy_target = False,               # adds noise to the target position
                reset_noise = 0.03,                 # [m], uniform noise in the first step 
                target_noise = 0.001,               # [m], uniform noise

                start_height = 0.06,                # [m], move to this height (above goal) after each reset
                insertion_height = 0.05,            # [m], height of beginning of insertion above goal
                insertion_indication_height = 0.04, # [m], height that indicates insertion

                limit_down_force = True,            # move up if fz >= z_force_max
                z_force_max = 3,                    # [N], Range; [-10, 10]
                z_step_up = 0.001,                 
                apply_down_force = True,            # move down if fz <= z_force_min
                z_force_min = 4,                    # [N], Range: [-10, 10]
                z_step_down = 0.001,

                round_forces = False,               # rounds forces down to decrease sensor noise
                switch_force_signs = False,         # switch force sign below insertion height
                indicator_threshold=0.002,          # [m], indicates a successful insertion
                
                **kwargs
        ):

        self.config = config[config_name];

        self.approach_target = approach_target
        self.RLoff = RLoff                     
        self.Human_off = Human_off                  
        self.lift_human_controller_goal = lift_human_controller_goal 

        self.human_z_off = human_z_off               
        self.finish_when_inserted = finish_when_inserted      

        self.random_reset = random_reset    # adds noise to the first step          
        self.noisy_target = noisy_target    # adds noise to the target position
        self.reset_noise = reset_noise      # [m], uniform noise in the first step
        self.target_noise = target_noise    # [m], uniform noise

        self.max_stepsize = max_stepsize
        self.human_ctrl_gain = human_ctrl_gain
        self.observation_type = observation_type
        self.action_type = action_type
        self.indicator_threshold=indicator_threshold

        self.insertion_indication_height = insertion_indication_height  # [m], height  that indicates insertion
        self.insertion_height = insertion_height                        # [m], depending on test object
        self.start_height = start_height                                # [m], begin the THT insertion with this height

        self.limit_down_force = limit_down_force
        self.apply_down_force = apply_down_force
        self.z_force_max = z_force_max
        self.z_force_min = z_force_min
        self.z_step_up = z_step_up
        self.z_step_down = z_step_down

        self.round_forces = round_forces
        self.switch_force_signs = switch_force_signs

        ## Modifiable settings ## 

        self.use_AR_tags = False        # use AR tag information in reward function
        self.use_sim = False            # use a policy trained on simulation in place of the human engineered controller

        self.debugging_mode = False     # random walk
        self.calibrationPeriod = 5      # seconds, time of force sensor calibration in the beginning

        self.safe_radius_max = 0.05     # around goal location
        self.safe_height_min = -0.05    # below goal location
        self.safe_height_max = 0.2      # above goal location
        
        self.calibration_steps = 1      # number of calibration steps after each initialization of the environment

        self.measureTime = False        # Flag for all timers
        self.printDiagnostics = True    # Flag for distance, reward and action prints

        # for TDM
        self.goal_dim_weights = [1, 1, 1]


        ## Unmodifiable initializations ## 

        if reward_type == 'usb_sparse':
            self.usb = USBDetector()
            self.usb._reset()
            self.usb_previously_in = False
        else:
            self.usb = None

        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        SawyerEnvBase.__init__(self, **kwargs)


        if self.action_mode=='torque':
            self.goal_space = self.config.TORQUE_SAFETY_BOX
        else:
            self.goal_space = self.config.POSITION_SAFETY_BOX
        self.reward_type = reward_type
        self.exact_state_goal = self.config.POSITION_GOAL_POS
        self._state_goal = self.config.POSITION_GOAL_POS

        #if reward_type == 'goal_image':
        self._image_goal = self.get_goal_img()
        self.reset_pos = self.config.POSITION_RESET_POS
        self.rate = rospy.Rate(self.config.UPDATE_HZ)

        self.u_rl = 0.0
        self.u_sim = 0.0
        self.u_h = 0.0
        self.r_d = 0.0
        self.r_l = 0.0
        self.r_f = 0.0
        self.r_i = 0.0
        self.r_a = 0.0
        self.r_u = 0.0
        self.r_img = 0.0
        self.r_falling = 0.0
        self.r_rotation = 0.0
        self.r_sliding_x = 0.0
        self.r_sliding_y = 0.0
        self.counter = 0
        self.pos_centered = np.array([0.0, 0.0, 0.0])
        self.goal_adjustment = np.array([0.0, 0.0, 0.0])
        self.force_centered = 0
        self.stationary_force = 0

        self.reset()
        self.stationary_force = self.force_sensor_calibration() # used to calibrate the force sensor in each reset
        self.start = timer()
        self.end = timer()


    @property
    def goal_dim(self):
        return 3

    def get_goal(self):
        goal = self._state_goal
        return goal

    def get_goal_img(self):
        img_path = '/home/ama/ros_ws/src/sawyer_control/src/sawyer_control/configs/goal_image/'
        target_img_path = img_path + 'target_image.png'

        img = Image.open(target_img_path)
        img.load()
        img = np.asarray( img, dtype="int32" )  # (32, 32)
        img = img.flatten()                     # (1024,), numpy array
        img = img / 255.0                   
        return img


    def compute_rewards(self, actions, obs):
        if self.reward_type=='sparse':
            r = 0.0
            if self.pos_centered[2] <= self.insertion_indication_height:
                r = 1.0
            return r

        if self.reward_type=='sparse_regularized':
            r = -np.linalg.norm(actions)
            if self.pos_centered[2] <= self.insertion_indication_height:
                r += 100.0
            return r

        if self.reward_type == 'goal_image':
            goal_img = self._image_goal
            curr_img = obs
            diff = goal_img - curr_img
            r_img = -np.linalg.norm(diff, ord=1) # 2-norm
            if self.printDiagnostics:
                print("r_img = ", r_img)
            return r_img

        if self.reward_type == 'usb_sparse':
            if self.usb.inserted:
                r = 1 
            else: 
                r = 0
            if self.printDiagnostics:
                print("ru = ", r)
            return r

        elif self.reward_type == 'hand_distance':
            distance = obs[:3]
            d_2 = np.linalg.norm(distance)
            r = - d_2
            if self.printDiagnostics:
                print("rd = ", r)
            return r 

        elif self.reward_type == 'hand_distance_z':
            distance = obs[2]
            r = - distance
            if self.printDiagnostics:
                print("rd = ", r)
            return r 

        elif self.reward_type == 'hand_distance_shaped':
            distance = obs[:3]
            d_2 = np.linalg.norm(distance)
            d_1 = np.linalg.norm(distance, ord=1)
            self.r_d = d_1
            self.r_f = obs[3]

            # feel free to include more rewards
            eps = np.exp(-700)
            self.r_l = np.log(d_2 + eps)
            self.r_i = 1/(d_2 + eps)
            self.r_a = np.linalg.norm(actions[0:2])

            c_d = -100.0    # scaling for distance term
            c_l = -0.0      # scaling for logarithmic term
            c_f = -0.1      # scaling for force term
            c_i =  0.002    # scaling for inverse distance
            c_a = -0.0      # scaling for action norm term 
            c_u =  0        # scaling for USB insertion

            rd = c_d * self.r_d
            rl = c_l * self.r_l
            rf = c_f * self.r_f 
            ri = c_i * self.r_i
            ra = c_a * self.r_a
            ru = c_u * self.r_u

            r = rd + rl + rf + ri + ra + ru
            if self.printDiagnostics:
                print("rd = ", rd, 'rl = ', rl, 'rf = ', rf , 'ri = ', ri, 'ra = ', ra, 'ru = ', ru)
            return r

        else: 
            raise NotImplementedError(
                '{} reward_type not supported.'.format(self.reward_type )
            )

    # simulation policy can be used instead of a P-controller
    def simulate_policy(self):
        file = "/home/ama/ros_ws/src/data/s3doodad/residualrl/sawyer-simulation-block-insertion-goal180/run0/id0/itr_300.pkl" # goal 1
        data = joblib.load(file)
        policy = data['policy']
        env = NormalizedBoxEnv(SimulationEnv180())
        path = rollout(
            env,
            policy,
            max_path_length=51, # must be greater than experiment path length 
        )
        return path['actions']

    def step(self, action):
        done = False

        if self.measureTime:
            self.end = timer()
            print('Time for one neural network evaluation [s] = ', self.end - self.start) # Time in seconds

        if self.debugging_mode:
            action = np.random.rand(3) * 2 - [1, 1, 1]
            action[2]  = -1
            print("In debugging mode.")

        if self.printDiagnostics:
            print("action = ", action)

        # difference to old version: action moves goal instead of defining an explicit, relative movement
        if not self.RLoff: 
            # TODO: Evaluate if z-movement can be constrained to [0, 0.01], max 1 cm up
            self.u_rl = action * self.max_stepsize
        else: 
            self.u_rl = np.array([0, 0, 0])

        if self.use_sim:
            self.u_sim = self.policy_action[self.counter][:3]
            self.u_h = self.u_sim
        else:
            observation = self._get_obs()
            obs = self.center_obs(observation)
            # move goal to tip of object to avoid crashes 
            if self.lift_human_controller_goal:
                obs = self.normalize_obs(obs)
            if self.Human_off:
                obs = np.array([0, 0, 0, 0])
            # TODO: set human controller goal slightly below insertion tip after insertion height has been reached [Gerrit]
            self.u_h = self.human_controller(obs[:3])

        if self.printDiagnostics:
            print("u_rl = ", self.u_rl, "u_h = ", self.u_h)


        theta = 5*np.pi/180.0
        axis = np.array([1, 0, 0]) # vertical = [1, 0, 0]

        q_curr = self._get_endeffector_pose()[3:] 
        q_rot = self.quaternion_from_euler(q_curr, axis, theta)

        ### EXECUTING ACTION
        u = self.u_h + self.u_rl

        # switching to stopping controller for downwards action
        self.action_service = self.action_service_alternate
        self._act(u, orientation = q_rot)
        self.action_service = self.action_service_main 

        ## POST ACTION ROUTINE 
        if self.apply_down_force or self.finish_when_inserted or self.limit_down_force:
            self.center_obs(self._get_obs())
            if self.apply_down_force:
                self.press_down_with_force()
            if self.pos_centered[2] <= self.insertion_indication_height and self.finish_when_inserted:
                self.reach_goal_with_tol(self.get_goal(), tol = 0.001, t = 5)
                rospy.sleep(1)
                done = True
            if self.limit_down_force:
                self.move_up_when_pressing_down()

        ## STAY IN SAFETY BOX
        self.safety_box()

        ### FINAL OBSERVATIONS AND REWARD CALCULATION
        observation = self.normalize_obs(self.center_obs(self._get_obs()))
        observation_centered = self.center_obs(observation)
        observation_normalized = self.normalize_obs(observation_centered)

        if self.observation_type == 'images' or self.reward_type == 'goal_image':
            img_obs = self._get_img_obs()

        if self.reward_type == 'goal_image':
            reward = self.compute_rewards(action, img_obs)
        else: 
            reward = self.compute_rewards(action, observation_centered)

        if self.printDiagnostics:
            np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
            print(
            "dist:", self.pos_centered,\
            " u:", u,\
            " rew:", "{:.5f}".format(reward),\
            " fz:", "{:.2f}".format(self.force_centered),
            )
            
        self.counter += 1
        info = self._get_info()

        if self.measureTime:
            self.end2 = timer()
            print('Time for one step [s] = ', self.end2 - self.start)
            self.start = timer()

        if self.observation_type == 'images':
            obs = img_obs
        else:
            obs = self.return_pos_obs(observation_normalized)

        return obs, reward, done, info


    def press_down_with_force(self):
        u_do = np.zeros(3)
        q_curr = self._get_endeffector_pose()[3:] 
        while self.force_centered <= self.z_force_min:
            u_do[2] -= self.z_step_down
            print('moving down', u_do, self.force_centered)
            self._act(u_do, orientation = q_curr)
            self.force_centered = self.center_obs(self._get_obs())[3]

    def move_up_when_pressing_down(self):
        u_up = np.zeros(3)
        q_curr = self._get_endeffector_pose()[3:] 
        while self.force_centered >= self.z_force_max:
            u_up[2] += self.z_step_up
            print('moving up', u_up, self.force_centered)
            self._act(u_up, orientation = q_curr)
            self.force_centered = self.center_obs(self._get_obs())[3]

    def safety_box(self):
        # Safety feature: Repelling fence
        if np.linalg.norm(self.pos_centered[0:2]) >= self.safe_radius_max or not self.safe_height_max >= self.pos_centered[2] >= self.safe_height_min: 
            u_ = -self.pos_centered * 0.5
            u_ = np.clip(u_, -0.01, 0.01)
            if self.safe_height_max >= self.pos_centered[2] >= self.safe_height_min:
                u_[2] = 0
            q_curr = self._get_endeffector_pose()[3:] 
            print("Out of safe zone, going back.") 
            self._act(u_, orientation = q_curr)

    def human_controller(self, pos):            
            # simple  clipped P-controller (in xyz), followed by a PID controller in joint space:
            u_h = -1 * pos
            # reduce z-gain of human controller
            u_h = u_h * self.human_ctrl_gain
            # option to reduce human controller step size
            d = 0.1 * np.ones(3)          # step size [m] - best so far: 0.005 
            u_h = np.clip(u_h, -d, d)     # smooth motion (small steps of length d or smaller)
            # turn off human controller in z-direction below certain height
            if self.pos_centered[2] <= self.start_height and self.human_z_off: 
                u_h[2] = 0
            return u_h


    def center_obs(self, observation):
        self.pos_centered = observation[:3] - self.get_goal() 
        self.force_centered = observation[3] - self.stationary_force

        if self.round_forces:
            self.round_force()
        if self.switch_force_signs:
            self.switch_force_sign()

        observation_centered = np.hstack((
            self.pos_centered,
            self.force_centered,
            ))

        return observation_centered

    def normalize_obs(self, observation_centered):
        observation_normalized = observation_centered - np.array([0, 0, self.insertion_height, 0]) # move coordinate sys origin to tip of object
        return observation_normalized

    def return_pos_obs(self, observation_normalized):
        if self.observation_type == 'position_xy':
            obs = observation_normalized[0:2]
        elif self.observation_type == 'position_xyz':
            obs = observation_normalized[0:3]
        elif self.observation_type == 'position_xyzf':
            obs = observation_normalized[0:4]
        else: 
            obs = observation_normalized
        return obs

    def round_force(self):
        self.force_centered = int(self.force_centered/1.0)*1 # round noisy force signal down 


    def switch_force_sign(self):
        if self.pos_centered[2] <= self.insertion_indication_height:
            self.force_centered = -self.force_centered


    def get_block_pose(self):
        block_id0_pose = self._get_ar_pos()['middle_marker_pose']
        block_id1_pose = self._get_ar_pos()['left_marker_pose']
        block_id2_pose = self._get_ar_pos()['right_marker_pose']
        blocks = np.hstack((
            block_id0_pose,
            block_id1_pose,
            block_id2_pose,
        ))
        return blocks

    def _get_obs(self):
        angles, velocities, endpoint_pose, endpoint_vel, endpoint_wrench = self.request_observation()
        obs = np.hstack((
            endpoint_pose[:3],      # end effector position: x-,y-,z-axes
            endpoint_wrench[2],     # end effector forces: z-axis
            ))
        return obs

    def _get_img_obs(self):
        img = self.get_image(grayscale = True, flatten = True) # size: 1024 
        return img 

    def _get_endeffector_pose(self):
        _, _, endpoint_pose, _, _ = self.request_observation()
        return endpoint_pose

    def _get_endeffector_velocities(self):
        _, _, _, endpoint_vel, _ = self.request_observation()
        linear_vel = endpoint_vel[:3]
        angular_vel = endpoint_vel[3:]
        return np.concatenate((linear_vel, angular_vel))

    def _get_joint_angles(self):
        angles, _, _, _, _ = self.request_observation()
        return angles

    def _get_endeffector_forces(self):
        _, _, _, _, endpoint_wrench = self.request_observation()
        return endpoint_wrench[0:3]

    def _get_endeffector_torques(self):
        _, _, _, _, endpoint_wrench = self.request_observation()
        return endpoint_wrench[3:6]

    def force_sensor_calibration(self):
        approach_target_original = self.approach_target
        action = np.zeros(self.action_space.shape)
        k = self.calibration_steps      # calibration repetitions, default = 3
        fz_mean = np.zeros(k)
        for j in range(k):
            cal_pos = self.get_goal() + np.array([0, 0, self.start_height])
            print('Moving to calibration position:', cal_pos)
            self.reach_goal_with_tol(cal_pos, tol = 0.001, t = 5)
            t_start = rospy.get_rostime()
            t_duration = rospy.Duration(self.calibrationPeriod) # we might want to use a longer calibration period
            now = t_start
            count = 1
            fz = 0.0
            print("Starting force sensor calibration")
            while now <= t_start + t_duration:
                fz += self._get_endeffector_forces()[2]
                now = rospy.get_rostime()
                count += 1
            fz_mean[j] = fz / count
            print("fz_mean = ", fz_mean[j],", calculated with ", count, " measurements.")
            self.reset()

        fz_mean = np.mean(fz_mean)
        print("Using the averaged value fz_mean = ", fz_mean)

        return fz_mean

    def sample_goal(self):
        self._state_goal = self.exact_state_goal + np.hstack([self.target_noise*2*(np.random.rand(2)-0.5), 0])

    def sample_reset(self):
        rp = self.reset_pos + np.hstack([self.reset_noise*2*(np.random.rand(2)-0.5), 0])
        return rp

    def reset(self):
        if self.printDiagnostics:
            print("Reset started")

        if self.noisy_target:
            self.sample_goal()

        if self.random_reset:
            curr_reset_pos = self.sample_reset()
        else:
            curr_reset_pos = self.reset_pos

        # multiple resets to ensure that the reset position is reached
        self.do_reset = True
        for i in range(2):
            if self.action_mode == "position":
                self._position_act(curr_reset_pos - self._get_endeffector_pose()[:3])
                print('resetting')
            else:
                self._reset_robot()          
        self.do_reset = False
        
        if self.use_AR_tags:
            input("Press Enter to continue after resetting all blocks ...")
        if self.use_sim:
            self.policy_action = self.simulate_policy()
            print("Policy loaded")

        if self.usb:
            self.usb_previously_in = False
            self.usb._reset()

        if self.printDiagnostics:
            print("Reset finished")
        
        # approach target 
        if self.approach_target:
            start_pos = self.get_goal() + (np.array(curr_reset_pos) - self.reset_pos) + np.array([0, 0, self.start_height])
            print('Moving to starting position:', start_pos)
            self.reach_goal_with_tol(start_pos, tol = 0.001, t = 5)

        if self.observation_type == 'images':
            obs = self._get_img_obs()
        else: 
            observation = self._get_obs()
            observation_centered = self.center_obs(observation)
            observation_normalized = self.normalize_obs(observation_centered)
            obs = self.return_pos_obs(observation_normalized)
            
        self.counter = 0
        return obs

    def _get_info(self):
        hand_distance = np.linalg.norm(self.pos_centered)
        hand_success=(hand_distance<self.indicator_threshold).astype(float)
        if self.reward_type == "usb_sparse":
            hand_success = np.array(self.usb.inserted).astype(float)
        return dict(
            hand_distance=hand_distance,
            hand_success=hand_success,
        )

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_success',
        ]:
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

    def convert_obs_to_goals(self, obs):
        return obs[:, 0:3]

    def _set_observation_space(self):
        if self.observation_type == 'images': 
            img_size = 32*32
            lows = np.zeros(img_size)
            highs = np.ones(img_size)

            self.observation_space = Box(
                lows,
                highs,
            )
        elif self.observation_type == 'position_xy':
            lows = np.hstack((
                self.config.POSITION_SAFETY_BOX_LOWS[0:2],
            ))
            highs = np.hstack((
                self.config.POSITION_SAFETY_BOX_HIGHS[0:2]
            ))

            self.observation_space = Box(
            lows,
            highs,
            )
        elif self.observation_type == 'position_xyz':
            lows = np.hstack((
                self.config.POSITION_SAFETY_BOX_LOWS,
            ))
            highs = np.hstack((
                self.config.POSITION_SAFETY_BOX_HIGHS,
            ))

            self.observation_space = Box(
            lows,
            highs,
            )
        elif self.observation_type == 'position_xyzf':
            lows = np.hstack((
                self.config.POSITION_SAFETY_BOX_LOWS,
                self.config.END_EFFECTOR_VALUE_LOW['force'],    # only forces in z-axis included
            ))
            highs = np.hstack((
                self.config.POSITION_SAFETY_BOX_HIGHS,
                self.config.END_EFFECTOR_VALUE_HIGH['force'],   # only z-axis included
            ))

            self.observation_space = Box(
            lows,
            highs,
            )
        else: 
            print('Observation type not supported.')


    def _set_action_space(self):
        if self.action_type == 'position_xy':
            self.action_space = Box(
                self.config.POSITION_CONTROL_LOW[0:2],
                self.config.POSITION_CONTROL_HIGH[0:2],
            )
        elif self.action_type == 'position_xyz':
            self.action_space = Box(
                self.config.POSITION_CONTROL_LOW[0:3],
                self.config.POSITION_CONTROL_HIGH[0:3],
            )
        else: 
            print('Action type not supported.')

    def set_to_goal(self, goal):
        self.reach_goal_with_tol(goal, tol = 0.001, t = 5)

    def reach_goal_with_tol(self, goal, tol = 0.001, t = 10, orientation = None): # choose larger t here
        time = rospy.get_time()  # in seconds
        finish_time = time + t  # in seconds
        err = goal - self._get_endeffector_pose()[:3]
        dist = dist = np.linalg.norm(err)
        self.action_service = self.action_service_alternate

        while dist > tol and time < finish_time:
            self._position_act(err, clip = False, orientation = None) 
            err = goal - self._get_endeffector_pose()[:3]
            dist = np.linalg.norm(err)
            print('error [m]: ', dist)
            time = rospy.get_time()
        self.action_service = self.action_service_main 
    
    def quaternion_from_euler(self, q_curr, axis, angle):
        q_rot = self.rotate_quaternion(q_curr, axis, angle)
        return q_rot

    def rotate_quaternion(self, q1, axis, angle):
        axis = np.array(axis)/np.linalg.norm(axis)
        q2 = np.hstack([np.cos(angle/2.0), np.sin(angle/2.0)*axis])
        nrm1 = np.linalg.norm(q1)
        nrm2 = np.linalg.norm(q2)
        if nrm1 != 1.0: q1 = q1/nrm1
        if nrm2 != 1.0: q2 = q2/nrm2
        q = self.hamilton_product(q1, q2)
        nrm = np.linalg.norm(q)
        if nrm != 1.0: q = q/nrm
        return q 

    def hamilton_product(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return w, x, y, z

