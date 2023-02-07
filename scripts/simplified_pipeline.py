import argparse, os, time
import cv2
import numpy as np
import polanalyser as pa


parser = argparse.ArgumentParser(description="Simplified pipeline for extracting false collored image representing the ")
parser.add_argument("image_dir", help="Directory where at least one raw image exists")
args = parser.parse_args()

if os.path.isdir(args.image_dir) is False:
    print("Directory", args.image_dir,"does not exists")
    quit(1)

print("start")
img_name = None
img_token = None
for file in os.listdir(args.image_dir):
    print(file)
    if "_raw.png" in file:
        img_name = file
        img_token = img_name[:-8] # remove '_raw.png' ending
    else:
        continue

    if img_name is None:
        print("raw image not found in", args.image_dir)
        quit(1)

    # Open image
    img_raw = cv2.imread(os.path.join(args.image_dir,img_name), cv2.IMREAD_GRAYSCALE)

    p_time = time.time()

    # Extract demosaiced rgb images (split + debayer)
    # s_time = time.time()
    demosaiced_color = pa.demosaicing(img_raw, pa.COLOR_PolarRGB)
    # e_time = time.time()
    # print(f"1,2: demosaic:\t {e_time-s_time:.3f} (s) -> {len(demosaiced_color)}x{demosaiced_color[0].shape} {demosaiced_color[0].dtype}")

    # Convert to monochrome images
    # s_time = time.time()
    demosaiced_mono = []
    for i in range(4):
        demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))
    # e_time = time.time()
    # print(f"3: grayscale:\t {e_time-s_time:.3f} (s) -> {len(demosaiced_mono)}x{demosaiced_mono[0].shape} {demosaiced_mono[0].dtype}")

    # Compute stokes parameters
    # s_time = time.time()
    radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    stokes_mono = pa.calcStokes(demosaiced_mono, radians)
    # e_time = time.time()
    # print(f"4: stokes:\t {e_time-s_time:.3f} (s) -> {stokes_mono.shape} {stokes_mono.dtype}")

    # Compute DoLP values
    # s_time = time.time()
    val_DoLP_mono  = pa.cvtStokesToDoLP(stokes_mono) # 0~1
    # e_time = time.time()
    # print(f"5: DoLP:\t {e_time-s_time:.3f} (s) -> {val_DoLP_mono.shape} {val_DoLP_mono.dtype}")

    # Compute AoLP values
    # s_time = time.time()
    val_AoLP_mono = pa.cvtStokesToAoLP(stokes_mono)
    # e_time = time.time()
    # print(f"6: AoLP:\t {e_time-s_time:.3f} (s) -> {val_AoLP_mono.shape} {val_AoLP_mono.dtype}")

    # Generate false-colored AoLP_DoLP representation
    # s_time = time.time()
    img_AoLP_DoLP_mono = pa.applyColorToAoLP(val_AoLP_mono, saturation=1.0, value=val_DoLP_mono)
    # e_time = time.time()
    # print(f"7: coloring:\t {e_time-s_time:.3f} (s) -> {img_AoLP_DoLP_mono.shape} {img_AoLP_DoLP_mono.dtype}")

    print(f"\nTotal time:\t {time.time()-p_time:.3f} (s)")

    cv2.imwrite(os.path.join(args.image_dir,img_token+"_pol.png"), img_AoLP_DoLP_mono)