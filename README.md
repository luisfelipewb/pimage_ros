# P Image Utils

ROS package with tools to process raw images from polarized cameras

Extracting raw images from bag file.
```
rosrun pimage_utils bat_to_raw.py input_bag output_dir image_topic
rosrun pimage_utils bag_to_raw.py  "/home/lwolfbat/bags/oil/2023-01-09-11-01-14.bag"  "/home/lwolfbat/bags/oil/img/" "/arena_camera_node/image_raw"
```

Convert raw images to polarized images
```
cd ws/src/pimage_utils/scripts/
raw_to_many.py ~/bags/oil/img/ --all #Super slow


Convert images to video
python3 img_to_vid.py ~/bags/oil/img/ fulltile