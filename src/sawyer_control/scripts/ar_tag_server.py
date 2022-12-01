#!/usr/bin/env python
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from sawyer_control.srv import *
import numpy as np
from tf.transformations import euler_from_quaternion
import math

class ARTracker():
    def __init__(self):
        # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)

        self.target_visible = False

        # Wait for the ar_pose_marker topic to become available
        rospy.loginfo("Waiting for ar_pose_marker topic...")
        rospy.wait_for_message('ar_pose_marker', AlvarMarkers) # ar_pose_marker must be running 

        # Subscribe to the ar_pose_marker topic to get the image width and height
        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.track_markers)

        rospy.loginfo("Marker messages detected. Starting follower...")


    def track_markers(self, msg):
        # Pick off the first marker (in case there is more than one)
        try:
            self.markers = msg.markers
            if not self.target_visible:
                rospy.loginfo("FOLLOWER is Tracking Target!")
            self.target_visible = True
            print("self.markers", self.markers)
        except:
            if self.target_visible:
                rospy.loginfo("FOLLOWER LOST Target!")
            self.target_visible = False

            return
    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        rospy.sleep(1)

def track_tags(unused):
    block_id0_marker_pose = np.array([0,0,0,0,0,0,0,0,0,0])
    block_id0_marker_roll = 0
    block_id0_marker_pitch = 0
    block_id0_marker_yaw = 0
    block_id1_marker_pose = np.array([0,0,0,0,0,0,0,0,0,0])
    block_id1_marker_roll = 0
    block_id1_marker_pitch = 0
    block_id1_marker_yaw = 0
    block_id2_marker_pose = np.array([0,0,0,0,0,0,0,0,0,0])
    block_id2_marker_roll = 0
    block_id2_marker_pitch = 0
    block_id2_marker_yaw = 0
    block_id3_marker_pose = np.array([0,0,0,0,0,0,0,0,0,0])
    block_id4_marker_pose = np.array([0,0,0,0,0,0,0,0,0,0])
    middle_marker_pose = np.array([0,0,0,0,0,0,0])
    left_marker_pose = np.array([0,0,0,0,0,0,0,0,0,0])
    right_marker_pose = np.array([0,0,0,0,0,0,0,0,0,0])
    left_marker_roll = 0
    left_marker_pitch = 0
    left_marker_yaw = 0
    right_marker_roll = 0
    right_marker_pitch = 0
    right_marker_yaw = 0

    for marker in ar.markers:
        pos = marker.pose.pose.position
        orientation = marker.pose.pose.orientation
        if marker.id == 0:
            block_id0_marker_quat = [orientation.x, orientation.y, orientation.z, orientation.w]
            (block_id0_marker_roll, block_id0_marker_pitch, block_id0_marker_yaw) = euler_from_quaternion(block_id0_marker_quat)
            block_id0_marker_pose = [pos.x, pos.y, pos.z, orientation.x, orientation.y, orientation.z, orientation.w, block_id0_marker_roll, block_id0_marker_pitch, block_id0_marker_yaw]
        if marker.id == 1:
            block_id1_marker_quat = [orientation.x, orientation.y, orientation.z, orientation.w]
            (block_id1_marker_roll, block_id1_marker_pitch, block_id1_marker_yaw) = euler_from_quaternion(block_id1_marker_quat)
            block_id1_marker_pose = [pos.x, pos.y, pos.z, orientation.x, orientation.y, orientation.z, orientation.w, block_id1_marker_roll, block_id1_marker_pitch, block_id1_marker_yaw]
        if marker.id == 2:
            block_id2_marker_quat = [orientation.x, orientation.y, orientation.z, orientation.w]
            (block_id2_marker_roll, block_id2_marker_pitch, block_id2_marker_yaw) = euler_from_quaternion(block_id2_marker_quat)
            block_id2_marker_pose = [pos.x, pos.y, pos.z, orientation.x, orientation.y, orientation.z, orientation.w, block_id2_marker_roll, block_id2_marker_pitch, block_id2_marker_yaw]
        #
        # if marker.id == 1:
        #     left_marker_quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        #     (left_marker_roll, left_marker_pitch, left_marker_yaw) = euler_from_quaternion(left_marker_quat)
        #     left_marker_pose = [pos.x, pos.y, pos.z, orientation.x, orientation.y, orientation.z, orientation.w, left_marker_roll, left_marker_pitch, left_marker_yaw]
        # if marker.id == 0:
        #     middle_marker_pose = [pos.x, pos.y, pos.z, orientation.x, orientation.y, orientation.z, orientation.w]
        # if marker.id == 2:
        #     right_marker_quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        #     (right_marker_roll, right_marker_pitch, right_marker_yaw) = euler_from_quaternion(right_marker_quat)
        #     right_marker_pose = [pos.x, pos.y, pos.z, orientation.x, orientation.y, orientation.z, orientation.w, right_marker_roll, right_marker_pitch, right_marker_yaw]
    # return ar_tagResponse(left_marker_pose, right_marker_pose, middle_marker_pose)
    return ar_tagResponse(block_id1_marker_pose, block_id2_marker_pose, block_id0_marker_pose)



def ar_tag_server():
    rospy.init_node("ar_tag_server", anonymous=True)
    global ar
    ar = ARTracker()
    s = rospy.Service('ar_tag', ar_tag, track_tags)
    rospy.spin()
if __name__ == '__main__':
    ar_tag_server()