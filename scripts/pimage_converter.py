#!/bin/python3

import rospy
from threading import Lock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import polanalyser as pa
import numpy as np

class PImageConverter(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        self.lock = Lock()

        self.radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Publishers
        self.pub_rgb = rospy.Publisher('~img_rgb', Image,queue_size=10)
        self.pub_pol = rospy.Publisher('~img_pol', Image,queue_size=10)

        # Subscribers
        rospy.Subscriber("/arena_camera_node/image_raw", Image,self.callback)

    def callback(self, msg):
        rospy.loginfo('Image received...')
        cv_img = self.br.imgmsg_to_cv2(msg)
        self.lock.acquire()
        self.image = cv_img
        self.lock.release()


    def convert_image(self, img_raw):
        # Demosaicing
        demosaiced_color = pa.demosaicing(img_raw, pa.COLOR_PolarRGB)

        # Extract regular RGB image (I_0 + I_90)
        img_rgb = np.empty((2048, 2448, 3), demosaiced_color[0].dtype)
        for i in range(3):
            img_0 = demosaiced_color[0][...,i] 
            img_90 = demosaiced_color[2][...,i]
            img_rgb[...,i] = cv2.addWeighted(img_0, 0.5, img_90, 0.5, 0.0)

        # Convert to monochrome images
        demosaiced_mono = []
        for i in range(4):
            demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))

        # Compute stokes parameters
        stokes_mono = pa.calcStokes(demosaiced_mono, self.radians)

        # Compute DoLP values
        val_DoLP_mono  = pa.cvtStokesToDoLP(stokes_mono) # 0~1

        # Compute AoLP values
        val_AoLP_mono = pa.cvtStokesToAoLP(stokes_mono)

        # Generate false-colored AoLP_DoLP representation
        img_AoLP_DoLP = pa.applyColorToAoLP(val_AoLP_mono, saturation=1.0, value=val_DoLP_mono)

        return img_rgb, img_AoLP_DoLP

    def start(self):
        #rospy.spin()
        while not rospy.is_shutdown():
            image = None

            # Copy image
            self.lock.acquire()
            if self.image is not None:
                image = self.image.copy()
                self.image = None
            self.lock.release()

            # Convert and publish
            if image is not None :
                img_rgb, img_pol = self.convert_image(image)
                msg_rgb = self.br.cv2_to_imgmsg(img_rgb, encoding='bgr8')
                msg_pol = self.br.cv2_to_imgmsg(img_pol, encoding='bgr8')
                self.pub_rgb.publish(msg_rgb)
                self.pub_pol.publish(msg_pol)
                rospy.loginfo('Images published')

            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("pimage_converter")
    my_node = PImageConverter()
    my_node.start()
