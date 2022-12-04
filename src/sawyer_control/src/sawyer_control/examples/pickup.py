from sawyer_control.envs.sawyer_grip_wrist_tilt_env import SawyerGripWristEnv
import numpy as np
import copy 
from pyquaternion import Quaternion

V_MAX = 5

def move_to_point(env, target, o):
    dp = target[:3]-o['observation'][:3]
    while np.linalg.norm(dp) > 0.01:
        u = np.zeros((6, ))
        u[:3] = np.clip(dp*50, -V_MAX, V_MAX)
        print(u)
        o, _, _, _ = env.step(u)
        dp = np.clip(target[:3]-o['observation'][:3], -V_MAX, V_MAX)
    return o

def grasp(env, bottle_pos, slot_pos=None):
    o = env.reset()
    import pdb; pdb.set_trace()
    grasp_target = np.append(bottle_pos, [0.15])

    waypoint = copy.deepcopy(o['observation'])[:3]
    waypoint[0] = 0.4
    o = move_to_point(env, waypoint, o)
    
    waypoint = copy.deepcopy(grasp_target)
    waypoint[0] = 0.4
    o = move_to_point(env, waypoint, o)

    o = move_to_point(env, grasp_target, o)

    u = np.zeros((6, ))
    u[3] = -1
    o, _, _, _ = env.step(u)




if __name__=='__main__':
    env = SawyerGripWristEnv(
    action_mode='position',
    config_name='charles_config',
    reset_free=False,
    position_action_scale=0.01,
    max_speed=0.4,
    step_sleep_time=0.2,
    )
    grasp(env, np.array([0.6207991443804394 ,  0.11184178415423862]))