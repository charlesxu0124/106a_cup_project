import time
from sawyer_control.envs.sawyer_grip_wrist_tilt_env import SawyerGripWristEnv
import numpy as np
import copy 

V_MAX = 5

def move_to_point(env, target, obs, grasp=False,):
    dp = target[:3]-obs['observation'][:3]
    while np.linalg.norm(dp) > 0.01:
        u = np.zeros((6, ))
        u[:3] = np.clip(dp*50, -V_MAX, V_MAX)
        u[3] = -1 if grasp else 1
        print(u)
        obs, _, _, _ = env.step(u)
        dp = np.clip(target[:3]-obs['observation'][:3], -V_MAX, V_MAX)
    return obs

def execute_pick_and_place(env, bottle_pos, slot_pos=None):
    o = env.reset()
    # import pdb; pdb.set_trace()
    grasp_target = np.append(bottle_pos, [0.14])
    waypoint = copy.deepcopy(grasp_target)
    waypoint[0] = 0.4
    o = move_to_point(env, waypoint, o)

    o = move_to_point(env, grasp_target, o)

    u = np.zeros((6, ))
    u[3] = -1
    o, _, _, _ = env.step(u)
    time.sleep(1.5)

    waypoint = copy.deepcopy(o['observation'][:3])
    waypoint[2] += 0.15
    o = move_to_point(env, waypoint, o, grasp=True)

    for _ in range(5):
        action = np.zeros((6,))
        action[3] = -1
        action[4] = 16/5
        o, _, _, _ = env.step(action)
        time.sleep(0.3)

    waypoint = copy.deepcopy(o['observation'][:3])
    waypoint[:2] = slot_pos
    o = move_to_point(env, waypoint, o, grasp=True)

    waypoint = copy.deepcopy(o['observation'][:3])
    waypoint[2] = 0.16
    o = move_to_point(env, waypoint, o, grasp=True)

    u = np.zeros((6, ))
    u[3] = 1
    o, _, _, _ = env.step(u)

    input('waiting to reset')

    waypoint = copy.deepcopy(o['observation'][:3])
    waypoint[2] = 0.25
    o = move_to_point(env, waypoint, o, grasp=False)

    for _ in range(5):
        action = np.zeros((6,))
        action[3] = 1
        action[4] = -16/5
        o, _, _, _ = env.step(action)
        time.sleep(0.3)

    waypoint = copy.deepcopy(o['observation'])[:3]
    waypoint[0] = 0.4
    o = move_to_point(env, waypoint, o)

    waypoint = np.array([0.4, 0.14, 0.18])
    o = move_to_point(env, waypoint, o)



if __name__=='__main__':
    env = SawyerGripWristEnv(
    action_mode='position',
    config_name='charles_config',
    reset_free=False,
    position_action_scale=0.01,
    max_speed=0.4,
    step_sleep_time=0.2,
    )
    execute_pick_and_place(env, bottle_pos=np.array([0.6207991443804394 ,  0.11184178415423862]),
    slot_pos=np.array([ 0.67055243, -0.10787934,]))