#!/usr/bin/env python
""" 
  Example code of how to convert ROS images to OpenCV's cv::Mat
  This is the solution to HW2, using Python.

  See also cv_bridge tutorials: 
    http://www.ros.org/wiki/cv_bridge
"""

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image

class ImageListener:

    def __init__(self):
        self.count = 0
        self.cv_bridge = CvBridge()

        # initialize a node
        rospy.init_node("image_listener")

        self.overlay_pub = rospy.Publisher('hand_demo_overlay_rgb', Image, queue_size=1)
        rgb_sub = message_filters.Subscriber('/camera/rgb/image_color', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image, queue_size=2)

        queue_size = 1
        # slop_seconds = 0.005
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

    def callback(self, rgb, depth):
        """ This is a callback which recieves images and processes them. """
        # convert image into openCV format
        bridge = CvBridge()
        try:
            # bgr8 is the pixel encoding -- 8 bits per color, organized as blue/green/red
            cv_image = self.cv_bridge.imgmsg_to_cv2(rgb, "bgr8")
            cv_depth = self.cv_bridge.imgmsg_to_cv2(depth, "32FC1")
            print cv_image.shape
            print cv_depth.shape
            print self.count
            self.count += 1
        except CvBridgeError, e:
            # all print statements should use a rospy.log_ form, don't print!
           rospy.loginfo("Conversion failed")

        overlay_msg = self.cv_bridge.cv2_to_imgmsg(cv_image)
        overlay_msg.header.stamp = rospy.Time.now()
        overlay_msg.header.frame_id = rgb.header.frame_id
        overlay_msg.encoding = 'bgr8'
        self.overlay_pub.publish(overlay_msg)

        # show the image
        # cv2.imshow("image_view", cv_image)
        # cv2.waitKey(0.01)


if __name__ == '__main__':
    listener = ImageListener()

    try:  
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()
