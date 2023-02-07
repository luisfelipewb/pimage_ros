import os
import cv2
import numpy as np
import argparse
import polanalyser as pa

def extract_from_raw(image_dir, flag_rgb, flag_mono, flag_dolp, flag_aolp, flag_aolpc, flag_aolpdoplp):
    """ Process raw images.
    """

    # Get a list of files containing the "_raw" tag
    raw_img_names = []
    for file in os.listdir(image_dir):
        if "_raw.png" in file:
            raw_img_names.append(file)
    raw_img_names.sort()
    
    # Process the raw images
    for raw_img_name in raw_img_names:
        token = raw_img_name[:-8] # remove _raw.png ending
        print("Processing image", token)

        # Read image
        raw = cv2.imread(os.path.join(image_dir, raw_img_name), cv2.IMREAD_GRAYSCALE)

        # RGB
        img_demosaiced = pa.demosaicing(raw, "COLOR_PolarRGB")
        (rgb_0, rgb_45, rgb_90, rgb_135) = (img_demosaiced[:,:,:,0], img_demosaiced[:,:,:,1], img_demosaiced[:,:,:,2], img_demosaiced[:,:,:,3])
        rgb = cv2.addWeighted(rgb_45, 0.5, rgb_135, 0.5, 0.0)

        # Convert from color to mono
        height, width, channels, polarizations = img_demosaiced.shape
        img_demosaiced_mono = np.empty((height, width, 1, 4), dtype=img_demosaiced.dtype)
        for i in range(polarizations):
            img_demosaiced_mono[:,:,0,i] = cv2.cvtColor(img_demosaiced[:,:,:,i], cv2.COLOR_BGR2GRAY)
        (mono_0, mono_45, mono_90, mono_135) = (img_demosaiced_mono[:,:,:,0], img_demosaiced_mono[:,:,:,1], img_demosaiced_mono[:,:,:,2], img_demosaiced_mono[:,:,:,3])
        mono = cv2.addWeighted(mono_45, 0.5, mono_135, 0.5, 0.0) # TODO Review how to merge multiple polarization channels

        # Calculate Stokes vector
        radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
        img_stokes = pa.calcStokes(img_demosaiced_mono, radians)

        # Calculate DoLP
        img_DoLP = pa.cvtStokesToDoLP(img_stokes) # 0~1
            
        # Calculate AoLP
        img_AoLP = pa.cvtStokesToAoLP(img_stokes) # 0~pi TODO:review represnetation

        # Apply collor
        img_AoLP_color = pa.applyColorToAoLP(img_AoLP) # Apply HSV collor map
        
        # Combine AoLP and DoLP
        img_AoLP_light = pa.applyColorToAoLP(img_AoLP, saturation=img_DoLP, value=1.0)
        img_AoLP_dark = pa.applyColorToAoLP(img_AoLP, saturation=1.0, value=img_DoLP)

        # Save image outputs        
        if flag_rgb:
            cv2.imwrite(os.path.join(image_dir, token+"_rgb.png"), rgb)
            cv2.imwrite(os.path.join(image_dir, token+"_rgb_0.png"), rgb_0)
            cv2.imwrite(os.path.join(image_dir, token+"_rgb_45.png"), rgb_45)
            cv2.imwrite(os.path.join(image_dir, token+"_rgb_90.png"), rgb_90)
            cv2.imwrite(os.path.join(image_dir, token+"_rgb_135.png"), rgb_135)
        if flag_mono:
            cv2.imwrite(os.path.join(image_dir, token+"_mono.png"), mono)
            cv2.imwrite(os.path.join(image_dir, token+"_mono_0.png"), mono_0)
            cv2.imwrite(os.path.join(image_dir, token+"_mono_45.png"), mono_45)
            cv2.imwrite(os.path.join(image_dir, token+"_mono_90.png"), mono_90)
            cv2.imwrite(os.path.join(image_dir, token+"_mono_135.png"), mono_135)
        if flag_dolp:
            cv2.imwrite(os.path.join(image_dir, token+"_dolp.png"), (img_DoLP * 255).round().astype(np.uint8))
        if flag_aolp:
            cv2.imwrite(os.path.join(image_dir, token+"_aolp.png"), (img_AoLP * 255).round().astype(np.uint8))
        if flag_aolpc:
            cv2.imwrite(os.path.join(image_dir, token+"_aolpc.png"), img_AoLP_color)
        if flag_aolpdoplp:
            cv2.imwrite(os.path.join(image_dir, token+"_aolpdolp_light.png"), img_AoLP_light)
            cv2.imwrite(os.path.join(image_dir, token+"_aolpdolp_dark.png"), img_AoLP_dark)
        
        
        tile = cv2.hconcat([rgb, img_AoLP_dark])

        cv2.imwrite(os.path.join(image_dir, token+"_tile.jpg"), tile)

        # # Full tile
        mono3 = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        mono3_90 = cv2.cvtColor(mono_90, cv2.COLOR_GRAY2BGR)
        img_DoLP3 = cv2.cvtColor((img_DoLP * 255).round().astype(np.uint8), cv2.COLOR_GRAY2BGR)

        top_tile = cv2.hconcat([mono3, rgb, img_DoLP3])
        bot_tile = cv2.hconcat([mono3_90, rgb_90, img_AoLP_dark])
        fulltile = cv2.vconcat([top_tile, bot_tile])
        cv2.imwrite(os.path.join(image_dir, token+"_fulltile.jpg"), fulltile)


        # # Explore
        # print(np.min(img_AoLP), np.max(img_AoLP))
        # cv2.imshow("aolpc",img_AoLP_color)
        # test = (img_AoLP / np.pi * 255).round().astype(np.uint8)
        # cv2.imshow("raw",raw)
        # cv2.imshow("test", test)
        # # cv2.imshow("aolp",img_AoLP)
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