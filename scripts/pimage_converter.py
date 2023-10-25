#!/bin/python3

import rospy
from threading import Lock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pimage_lib import pimage as pi
import cv2
import datetime
import os
from pathlib import Path

class PImageConverter(object):
    def __init__(self):
        # Params
        self.rate = rospy.get_param('~rate', 10.0)
        self.store_raw = rospy.get_param('~store_raw', False)
        self.store_rgb = rospy.get_param('~store_rgb', False)
        self.store_pol = rospy.get_param('~store_pol', False)

        if self.store_pol or self.store_rgb or self.store_raw:
            # generate output folder name based on date and time
            self.output_folder = rospy.get_param('~output_folder', '~/')
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            self.output_folder = f'{self.output_folder}/{date_str}/'

            # create a folder if it does not exist
            rospy.loginfo(f"Output folder: {self.output_folder}")
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # Variables
        self.loop_rate = rospy.Rate(self.rate)
        self.last_image = None
        self.last_header = None
        self.new_image_flag = False
        self.br = CvBridge()
        self.lock = Lock()

        # Publishers
        self.pub_rgb = rospy.Publisher('~img_rgb', Image, queue_size=10)
        self.pub_pol = rospy.Publisher('~img_pol', Image, queue_size=10)

        # Subscribers
        rospy.Subscriber("image_raw", Image,self.raw_image_cb)

    def raw_image_cb(self, msg):
        # rospy.loginfo('Image received...')
        cv_img = self.br.imgmsg_to_cv2(msg)
        self.lock.acquire()
        self.last_header = msg.header
        self.last_image = cv_img
        self.new_image_flag = True
        self.lock.release()

    def start(self):
        while not rospy.is_shutdown():
            image = None

            # Copy image
            self.lock.acquire()
            if self.new_image_flag:
                image = self.last_image.copy()
                header = self.last_header
                self.lock.release()

                # Convert and publish
                img_rgb, img_pol = pi.extractColorAndPol(image)

                msg_rgb = self.br.cv2_to_imgmsg(img_rgb, encoding='bgr8')
                msg_pol = self.br.cv2_to_imgmsg(img_pol, encoding='bgr8')

                msg_rgb.header = header
                msg_pol.header = header

                self.pub_rgb.publish(msg_rgb)
                self.pub_pol.publish(msg_pol)

                # Store images
                if self.store_raw:
                    output_path = self.output_folder + f'frame{header.seq:06d}_raw.png'
                    rospy.loginfo(f'Saving raw image to {output_path}')
                    cv2.imwrite(output_path, image)
                if self.store_rgb:
                    output_path = self.output_folder + f'frame{header.seq:06d}_rgb.png'
                    rospy.loginfo(f'Saving rgb image to {output_path}')
                    cv2.imwrite(output_path, img_rgb)
                if self.store_pol:
                    output_path = self.output_folder + f'frame{header.seq:06d}_pol.png'
                    rospy.loginfo(f'Saving pol image to {output_path}')
                    cv2.imwrite(output_path, img_pol)

            else:
                self.lock.release()

            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("pimage_converter")
    my_node = PImageConverter()
    rospy.loginfo('Starting pimage_converter node')
    my_node.start()
