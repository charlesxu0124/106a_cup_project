from image import segment_cup, get_target


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
    Checkerboard H 23.5cm / 9 squares
                W 18.3cm / 7 squares
        rosrun camera_calibration cameracalibrator.py --size 9x7 --square 0.026 image:=/usb_cam/image_raw camera:=/usb_cam
    input: tuple of pixels
    returns: x, y, z coordinates of grasp point for FK planner
    '''
    camera_to_robot = # result of calibration, load from TF
    x,y = camera_to_robot @ pixel_to_camera @ pixels 

def planner():
    return None
# planner goes to the coords


if __name__ == 'main':
    # take pic
    pixel_coords = segment_cup(img=None)
    grasp_point = pixel_to_robot(pixel_coords)
    place_point = get_target(img=None)
    planner() # Assigned to: Charles
