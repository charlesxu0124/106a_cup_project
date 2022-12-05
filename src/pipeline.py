# from image import segment_cup, get_target
import numpy as np
from sawyer_control.src.sawyer_control.envs.sawyer_grip_wrist_tilt_env import SawyerGripWristEnv
from sawyer_control.src.sawyer_control.examples.pickup import execute_pick_and_place
from sawyer_control.src.sawyer_control.helper.video_capture import VideoCapture

K = np.array([[604.026546, 0.000000, 331.477939],
[0.000000, 602.778325, 214.438981],
[0.000000, 0.000000, 1.000000]])

# select grasp point in pixels -> (538, 495)
# In image:
# def segment_cup(img):
#     '''
#     Assigned to: Francis, Vincy
#     returns tuple of pixels of cup center
#     '''

# def get_target(img):
#     '''
#     Assigned to: Francis, Vincy
#     returns x,y,z of center of place point
#     '''


def pixel_to_robot(pixels):
    '''
    Assigned to: Allie, Charles. Soft ddl 11/27
    Calibration info: 
    Checkerboard H 23.5cm / 8 squares
                W 18.3cm / 6 squares
        rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.026 image:=/usb_cam/image_raw camera:=/usb_cam
    input: tuple of pixels
    returns: x, y, z coordinates of grasp point for FK planner

    curr: 0.722921332930742 ,  0.12386950470971811
    target: ([0.56342208,      0.04239593, 

    pred: 0.6815324235101948 ,  -0.06193677822190522
    actual: 0.53706831, -0.08917044

    pred: 0.797421369887727 ,  0.09732575000520052
    actual: 0.617203  , 0.02721379,

    base_link og z: 0.90415704
 

    '''
    pixels = np.append(pixels, [1])
    # Extrinsics
    robot_to_camera = np.zeros((4,4))
    robot_to_camera_R = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    robot_to_camera_p = np.array([ 0.63438559, -0.03667158,  0.79])
    robot_to_camera[:3,:3] = robot_to_camera_R
    robot_to_camera[:3,3] = robot_to_camera_p
    robot_to_camera[3,3] = 1 

    print("extrinsics",robot_to_camera)

    camera_to_robot = np.linalg.inv(robot_to_camera)
    pixel_to_camera = np.append(np.linalg.inv(K) @ pixels, [1])
    # Intrinsics
    x,y, _, _ = camera_to_robot @ pixel_to_camera
    offset_x, offset_y = 0.17, 0.09
    return x-offset_x,y-offset_y


def get_target(img=None):
    '''
    Returns: coordinate of any open slot in spatial frame
    '''

    pos = dict(1=np.array([ 0.66586983, -0.20091158,]),
            2 = np.array([ 0.56408536, -0.19261755]),
            3=np. array([ 0.47533491, -0.18826577,]),
            5=np.array([ 0.55825764, -0.0977271 ,]),
            4=np. array([ 0.67055243, -0.10787934,]),
            6=np.array([ 0.47370866, -0.09306668,]), 
            7=np.array([ 0.6702069 , -0.02252326,]),
            8=np.array([ 0.56647497, -0.00757453,]),
            9 = np.array([ 0.47168967, -0.00487166]))


if __name__ == 'main':
    env = SawyerGripWristEnv(
    action_mode='position',
    config_name='charles_config',
    reset_free=False,
    position_action_scale=0.01,
    max_speed=0.4,
    step_sleep_time=0.2,
    )
    # take pic
    resource = "/dev/video0"
    cap = VideoCapture(resource)
    image = cap.read()
    pixel_coords = segment_cup(img=None)
    grasp_point = pixel_to_robot(pixel_coords)
    place_point = get_target(img=None)
    execute_pick_and_place(env, bottle_pos=grasp_point, slot_pos=place_point)
