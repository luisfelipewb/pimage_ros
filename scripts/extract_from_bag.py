#!/bin/python3

import os
import cv2
import argparse

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def bag_to_raw(input_bag, output_dir, image_topic):
    """Extract raw images from a rosbag
    """

    bag = rosbag.Bag(input_bag, "r")
    bridge = CvBridge()
    count = 0
    info = bag.get_type_and_topic_info()
    total_msgs = info.topics[image_topic].message_count
    for _, msg, _ in bag.read_messages(topics=[image_topic]):

        cv_img_raw = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # print("raw shape", cv_img_raw.shape)
        out_path = os.path.join(output_dir, "frame{:06}_raw.png".format(msg.header.seq))
        cv2.imwrite(out_path, cv_img_raw)

        count += 1
        if count%10 == 0:
            print(f"Processing {100*count/total_msgs:.1f}%", end="\r")

    bag.close()
    print(f"Finished! Extracted {count}/{total_msgs} images")

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("input_bag", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.", default='/arena_camera_node/image_raw')

    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir)
        print("Creating directory", args.output_dir)

    except FileExistsError:
        pass

    print(f"Extract images from {args.input_bag} on topic {args.image_topic} into {args.output_dir}" )

    bag_to_raw(args.input_bag, args.output_dir, args.image_topic)
