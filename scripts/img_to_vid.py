import os
import cv2
import numpy as np
import argparse
import numpy as np


def img_to_vid(image_dir, tag):
    """Create video.
    """

    # Get a list of files containing the "_raw" tag
    img_names = []
    for file in os.listdir(image_dir):
        if tag+'.png' in file:
            img_names.append(file)
    
    img_names.sort()
    # Process the raw images
    # print("Found", img_names.size, "images")

    # Prepare video
    frameSize = (2448//4, 2048//4)
    # frameSize = (2048, 2448)
    video_name = os.path.join(image_dir, 'video_'+tag+'.mp4')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter(video_name, fourcc, 20.0, frameSize)
    out = cv2.VideoWriter(video_name, fourcc, 5.0, frameSize)



    for img_name in img_names:

        print("Processing image", img_name)
        
        image = cv2.imread(os.path.join(image_dir, img_name))
        # print(np.shape(image))
        # print(frameSize)
        frame = cv2.resize(image, frameSize)
        
        cv2.putText(frame, img_name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow("image", frame)
        # cv2.waitKey(1)

        out.write(frame)

    out.release()
        
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract data from raw images.")

    parser.add_argument("image_dir", help="Image directory.")
    parser.add_argument("tag", help="Tag for selecting the images.")
    
    args = parser.parse_args()

    if os.path.isdir(args.image_dir) is False:
        print("Directory", args.dir,"does not exists")
        quit()
    
    img_to_vid(args.image_dir, args.tag)
    quit()