import os
import cv2
import numpy as np
import argparse
import polanalyser as pa

def extract_from_raw(image_dir, flag_rgb, flag_mono, flag_dolp, flag_aolp, flag_aolpc, flag_aolpdoplp):
    """Extract a folder of images from a rosbag.
    """

    # Get a list of files containing the "_raw" tag
    raw_img_names = []
    for file in os.listdir(image_dir):
        if "_raw.png" in file:
            raw_img_names.append(file)
    
    # Process the raw images
    for raw_img_name in raw_img_names:
        token = raw_img_name[:-8] # remove _raw.png ending
        print("Processing image", token)
        
        image = cv2.imread(os.path.join(image_dir, raw_img_name))
        # Reduce from 3 channes with same value (mono)
        raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print("raw shape", raw.shape)

        img_demosaiced = pa.demosaicing(raw, "COLOR_PolarRGB")
        (rgb_0, rgb_45, rgb_90, rgb_135) = (img_demosaiced[:,:,:,0], img_demosaiced[:,:,:,1], img_demosaiced[:,:,:,2], img_demosaiced[:,:,:,3])

        if flag_rgb:
            rgb = cv2.addWeighted(rgb_0, 0.5, rgb_90, 0.5, 0.0)
            cv2.imwrite(os.path.join(image_dir, token+"_rgb.png"), rgb)


        # Convert from color to mono
        height, width, channels, polarizations = img_demosaiced.shape
        img_demosaiced_mono = np.empty((height, width, 1, 4), dtype=img_demosaiced.dtype)
        for i in range(polarizations):
            img_demosaiced_mono[:,:,0,i] = cv2.cvtColor(img_demosaiced[:,:,:,i], cv2.COLOR_BGR2GRAY)
        (mono_0, mono_45, mono_90, mono_135) = (img_demosaiced_mono[:,:,:,0], img_demosaiced_mono[:,:,:,1], img_demosaiced_mono[:,:,:,2], img_demosaiced_mono[:,:,:,3])

        if flag_mono:
            mono = cv2.addWeighted(mono_0, 0.5, mono_90, 0.5, 0.0) # TODO Review how to merge multiple polarization channels
            cv2.imwrite(os.path.join(image_dir, token+"_mono.png"), mono)
        # Calculate Stokes vector
        radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
        img_stokes = pa.calcStokes(img_demosaiced_mono, radians)

        # Calculate DoLP
        img_DoLP = pa.cvtStokesToDoLP(img_stokes) # 0~1
        if flag_dolp:
            # cv2.imshow("img_DoLP", img_DoLP.astype("uint8"))
            # print(np.max(img_DoLP))
            cv2.imwrite(os.path.join(image_dir, token+"_dolp.png"), (img_DoLP * 255).round().astype(np.uint8))
            
        # Calculate AoLP
        img_AoLP = pa.cvtStokesToAoLP(img_stokes) # 0~pi TODO:review represnetation
        if flag_aolp:
            cv2.imwrite(os.path.join(image_dir, token+"_aolp.png"), (img_AoLP * 255).round().astype(np.uint8))
            # cv2.imshow("img_AoLP", img_AoLP)
            # cv2.waitKey()

        # Apply collor
        print(np.shape(img_AoLP))
        img_AoLP_color = pa.applyColorToAoLP(img_AoLP) # Apply HSV collor map
        if flag_aolpc:
            cv2.imwrite(os.path.join(image_dir, token+"_aolpc.png"), img_AoLP_color)
            # cv2.imshow("img_AoLP_color", img_AoLP_color)
            # cv2.waitKey()

        # Combine with DoLP
        img_AoLP_light = pa.applyColorToAoLP(img_AoLP, saturation=img_DoLP, value=1.0)
        img_AoLP_dark = pa.applyColorToAoLP(img_AoLP, saturation=1.0, value=img_DoLP)
        if flag_aolpdoplp:
            cv2.imwrite(os.path.join(image_dir, token+"_aolpdolp.png"), img_AoLP_light)
            cv2.imwrite(os.path.join(image_dir, token+"_aolpdolp.png"), img_AoLP_dark)
            # cv2.waitKey()
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract data from raw images.")

    parser.add_argument("image_dir", help="Image directory.")
    parser.add_argument("--rgb", action='store_true', default=False, help="Store rgb png images")
    parser.add_argument("--mono", action='store_true', default=False, help="Store Grayscale png images")
    parser.add_argument("--dolp", action='store_true', default=False, help="Store Degree of Polarization png images")
    parser.add_argument("--aolp", action='store_true', default=False, help="Store Angle of Polarization png images")
    parser.add_argument("--aolpc", action='store_true', default=False, help="Store Angle of Polarization png images")
    parser.add_argument("--aolpdolp", action='store_true', default=False, help="Store Angle of Polarization png images")
    parser.add_argument("--all", action='store_true', default=False, help="All outputs")

    
    args = parser.parse_args()

    if os.path.isdir(args.image_dir) is False:
        print("Directory", args.dir,"does not exists")
        quit()

    if args.all:
        extract_from_raw(args.image_dir, True, True, True, True, True, True)
    else:
        extract_from_raw(args.image_dir, args.rgb, args.mono, args.dolp, args.aolp, args.aolpc, args.aolpdolp)
    quit()