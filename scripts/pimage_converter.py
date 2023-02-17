#!/bin/python3

import rospy
from threading import Lock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pimage_lib import pimage as pi

class PImageConverter(object):
    def __init__(self):
        # Params
        self.image = None
        self.new_image = False
        self.br = CvBridge()
        self.lock = Lock()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # Publishers
        self.pub_rgb = rospy.Publisher('~img_rgb', Image,queue_size=10)
        self.pub_pol = rospy.Publisher('~img_pol', Image,queue_size=10)

        # Subscribers
        rospy.Subscriber("image_raw", Image,self.callback)

    def callback(self, msg):
        # rospy.loginfo('Image received...')
        cv_img = self.br.imgmsg_to_cv2(msg)
        self.lock.acquire()
        self.image = cv_img
        self.new_image = True
        self.lock.release()

    def start(self):
        while not rospy.is_shutdown():
            image = None

            # Copy image
            self.lock.acquire()
            if self.new_image:
                image = self.image.copy()
                self.lock.release()

                # Convert and publish
                img_rgb, img_pol = pi.extractColorAndPol(image)
                msg_rgb = self.br.cv2_to_imgmsg(img_rgb, encoding='bgr8')
                msg_pol = self.br.cv2_to_imgmsg(img_pol, encoding='bgr8')
                self.pub_rgb.publish(msg_rgb)
                self.pub_pol.publish(msg_pol)
                # rospy.loginfo('Images published')

            else:
                self.lock.release()

            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("pimage_converter")
    my_node = PImageConverter()
    my_node.start()
