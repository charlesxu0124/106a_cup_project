#!/usr/bin/env python
from sawyer_control.srv import *
import rospy
from sensor_msgs.msg import Image as Image_msg
from sawyer_control.msg import imagemsg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import copy
import thread
import numpy as np

import intera_interface
import time

import cv2

class Latest_observation(object):
    def __init__(self):
        # color image:
        self.img_cv2 = None
        self.img_cropped = None
        self.img_msg = None

        # depth image:
        self.d_img_raw_npy = None  # 16 bit raw data
        self.d_img_cropped_npy = None
        self.d_img_cropped_8bit = None
        self.d_img_msg = None

class KinectRecorder(object):
    def __init__(self):
        self.flat_img_publisher = rospy.Publisher("/flat_robot_image", imagemsg, queue_size=10)
        rospy.Subscriber("/kinect2/hd/image_color", Image_msg, self.store_latest_image)

        self.ltob = Latest_observation()
        self.ltob_aux1 = Latest_observation()

        self.bridge = CvBridge()

        def spin_thread():
            rospy.spin()

        thread.start_new(spin_thread, ())

    def store_latest_image(self, data):
        # rospy.logerr("storing")
        self.ltob.img_msg = data
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (1920, 1080)
        self.ltob.img_cv2 = self.crop_highres(cv_image) #(1000, 1000)

        img = np.array(self.ltob.img_cv2)
        # rospy.logerr(img.shape)
        # img = img[::2, ::2, :]
        # rospy.logerr(img.shape)
        flat_img = img[...,::-1].flatten().tolist()
        self.flat_img_publisher.publish(imagemsg(flat_img))
        self.last_image_timestamp = time.time()

    # def crop_highres(self, cv_image):
    #     startcol = 180
    #     startrow = 0
    #     endcol = startcol + 1500
    #     endrow = startrow + 1500
    #     cv_image = copy.deepcopy(cv_image[startrow:endrow, startcol:endcol])
    #     cv_image = cv2.resize(cv_image, (0, 0), fx=0.66666666666, fy=0.925925926, interpolation=cv2.INTER_AREA)
    #     return cv_image

    def crop_highres(self, cv_image):
        startx = 432
        starty = 24
        endx = startx + 1056
        endy = starty + 1056
        cv_image = copy.deepcopy(cv_image[starty:endy:22, startx:endx:22])
        # cv_image = copy.deepcopy(cv_image[startrow:endrow, startcol:endcol])
        # cv_image = cv2.resize(cv_image, (0, 0), fx=0.66666666666, fy=0.925925926, interpolation=cv2.INTER_AREA)
        return cv_image


class SawyerRecorder(object):
    def __init__(self):
        self.flat_img_publisher = rospy.Publisher("/flat_robot_image", imagemsg, queue_size=10)
        self.last_image_timestamp = time.time()

        self.sawyer_cameras = intera_interface.Cameras()
        self.bridge = CvBridge()

        if not self.sawyer_cameras.verify_camera_exists("head_camera"):
            rospy.logerr("Could not detect the head_camera.")

        rospy.loginfo("Opening camera")
        self.sawyer_cameras.start_streaming("head_camera")
        self.sawyer_cameras.set_callback("head_camera", self.store_latest_image, rectify_image=True, queue_size=10)

        self.ltob = Latest_observation()
        self.ltob_aux1 = Latest_observation()

        def spin_thread():
            rospy.spin()

        thread.start_new(spin_thread, ())

    def store_latest_image(self, data):
        # rospy.logerr(("image subscriber elapsed time", time.time() - self.last_image_timestamp))
        self.ltob.img_msg = data
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (800, 1280, 3)
        self.ltob.img_cv2 = self.crop_highres(cv_image) #(1000, 1000)

        img = np.array(self.ltob.img_cv2)
        flat_img = img.flatten().tolist()
        self.flat_img_publisher.publish(imagemsg(flat_img))
        self.last_image_timestamp = time.time()

    def crop_highres(self, cv_image):
        startx = 500
        starty = 350
        endx = startx + 500
        endy = starty + 300
        cv_image = copy.deepcopy(cv_image[starty:endy, startx:endx])
        # cv_image = copy.deepcopy(cv_image[startrow:endrow, startcol:endcol])
        # cv_image = cv2.resize(cv_image, (0, 0), fx=0.66666666666, fy=0.925925926, interpolation=cv2.INTER_AREA)
        return cv_image


class USBRecorder(object):
    def __init__(self):
        self.flat_img_publisher = rospy.Publisher("/flat_robot_image", imagemsg, queue_size=10)
        self.last_image_timestamp = time.time()

        # self.sawyer_cameras = intera_interface.Cameras()
        # self.bridge = CvBridge()

        # if not self.sawyer_cameras.verify_camera_exists("head_camera"):
        #     rospy.logerr("Could not detect the head_camera.")

        # rospy.loginfo("Opening camera")
        # self.sawyer_cameras.start_streaming("head_camera")
        # self.sawyer_cameras.set_callback("head_camera", self.store_latest_image, rectify_image=True, )

        self.ltob = Latest_observation()
        self.ltob_aux1 = Latest_observation()

        # def spin_thread():
        #     rospy.spin()

        # thread.start_new(spin_thread, ())

    def store_latest_image(self, data):
        rospy.logerr(("image subscriber elapsed time", time.time() - self.last_image_timestamp))
        self.ltob.img_msg = data
        # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (800, 1280, 3)
        # self.ltob.img_cv2 = self.crop_highres(cv_image) #(1000, 1000)

        # img = np.array(self.ltob.img_cv2)
        # flat_img = img.flatten().tolist()
        # self.flat_img_publisher.publish(imagemsg(flat_img))
        self.last_image_timestamp = time.time()

    def crop_highres(self, cv_image):
        startx = 500
        starty = 350
        endx = startx + 500
        endy = starty + 300
        cv_image = copy.deepcopy(cv_image[starty:endy, startx:endx])
        # cv_image = copy.deepcopy(cv_image[startrow:endrow, startcol:endcol])
        # cv_image = cv2.resize(cv_image, (0, 0), fx=0.66666666666, fy=0.925925926, interpolation=cv2.INTER_AREA)
        return cv_image


class USBImage(object):
    def __init__(self):
        self.flat_img_publisher = rospy.Publisher("/flat_robot_image", imagemsg, queue_size=10)
        self.last_image_timestamp = time.time()

        self.ltob = Latest_observation()

        resource = "/dev/video0"
        self.cap = cv2.VideoCapture(resource)
        if not cap.isOpened():
            print "Error opening resource: " + str(resource)
            print "Maybe opencv VideoCapture can't open it"
            exit(0)

    def run(self):
        rval, frame = cap.read()
        while rval:
            rval, frame = cap.read()
            self.store_latest_image(frame)

    def store_latest_image(self, data):
        rospy.logerr(("image subscriber elapsed time", time.time() - self.last_image_timestamp))
        self.ltob.img_msg = data
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (800, 1280, 3)
        self.ltob.img_cv2 = self.crop_highres(cv_image) #(1000, 1000)

        img = np.array(self.ltob.img_cv2)
        flat_img = img.flatten().tolist()
        self.flat_img_publisher.publish(imagemsg(flat_img))
        self.last_image_timestamp = time.time()

    def crop_highres(self, cv_image):
        startx = 500
        starty = 350
        endx = startx + 500
        endy = starty + 300
        cv_image = copy.deepcopy(cv_image[starty:endy, startx:endx])
        # cv_image = copy.deepcopy(cv_image[startrow:endrow, startcol:endcol])
        # cv_image = cv2.resize(cv_image, (0, 0), fx=0.66666666666, fy=0.925925926, interpolation=cv2.INTER_AREA)
        return cv_image


def get_observation(unused):
    img = kr.ltob.img_cv2
    img = np.array(img)
    image = img.flatten().tolist()
    return imageResponse(image)

def image_server():
    s = rospy.Service('images', image, get_observation)
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node('image_server', anonymous=True)
    kr = KinectRecorder()
    # kr = SawyerRecorder()
    # kr = USBRecorder()
    image_server()

