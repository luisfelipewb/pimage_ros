import os, time
import cv2
import numpy as np
from pimage_lib import pimage as pimg

img_dir = "../img/"
img_file = "frame00000_raw.png"
img_path = os.path.join(img_dir, img_file)
print(f"Opening image {img_path}")

# Open image
img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Track total time
# p_time = time.time()

# Extract demosaiced rgb images: 4x(2048, 2448, 3) uint8
demosaiced_color = pimg.demosaicing(img_raw)

# cv2.imshow("0", demosaiced_color[0])
# cv2.imshow("1", demosaiced_color[1])
# cv2.imshow("2", demosaiced_color[2])
# cv2.imshow("3", demosaiced_color[3])

# Extract monochorme polarization channels
demosaiced_mono = []
for i in range(4):
    demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))


# Extract regular RGB image (I_0 + I_90)
# img_rgb = np.empty((2048, 2448, 3), demosaiced_color[0].dtype)
# for i in range(3):
#     img_0 = demosaiced_color[0][...,i] 
#     img_90 = demosaiced_color[2][...,i]
img_rgb = pimg.rgb(demosaiced_color)
# cv2.imshow("img_rgb", img_rgb)


# Same as regular filter
# img_rgb_90 = demosaiced_color[2]

# Monochrome
# img_mono = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Compute stokes parameters for each color: (2048, 2448, 3, 3) float64
# radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
# stokes_color = pa.calcStokes(demosaiced_color, radians)

# Compute stokes parameters for monochrome: (2048, 2448, 3) float64
# radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
# stokes_mono = pa.calcStokes(demosaiced_mono, radians)
stokes_mono = pimg.calcStokes(demosaiced_mono)
print(f"Shape {np.shape(stokes_mono)}")
print(f"max: {np.max(stokes_mono)}")
print(f"min: {np.min(stokes_mono)}")


# Compute DoLP values for each color: (2048, 2448, 3) float64
# val_DoLP_color  = pa.cvtStokesToDoLP(stokes_color) # 0~1

# Compute DoLP values for monochrome: (2048, 2448, 1) float64
val_DoLP_mono  = pimg.calcDoLP(stokes_mono) # 0~1
print(f"max: {np.max(val_DoLP_mono)}")
print(f"min: {np.min(val_DoLP_mono)}")

# Generate DoLP image: (2048, 2448, 3) uint8
# img_DoLP_color = (val_DoLP_color * 255).round().astype(np.uint8)

img_DoLP_mono = (val_DoLP_mono * 255).round().astype(np.uint8)
cv2.imshow("img_DoLP_mono", img_DoLP_mono)
cv2.waitKey()

# Compute AoLP values: (2048, 2448, 3) float64
# val_AoLP_color = pa.cvtStokesToAoLP(stokes_color)

# Compute AoLP values: (2048, 2448, 1) float64
# val_AoLP_mono = pa.cvtStokesToAoLP(stokes_mono)

# Generate False colored AoLP_DoLP representation for all three color channels
# img_AoLP_DoLP_color = np.empty(stokes_color.shape, dtype=img_DoLP_color.dtype)
# for i in range(3):
    # img_AoLP_DoLP_color[..., i] = pa.applyColorToAoLP(val_AoLP_color[:,:,i], saturation=1.0, value=val_DoLP_color[:,:,i])

# Generate False colored AoLP_DoLP representation for monochrome
# img_AoLP_DoLP_mono = pa.applyColorToAoLP(val_AoLP_mono, saturation=1.0, value=val_DoLP_mono)


# print(f"Total time:\t{time.time()-p_time:.3f}")


# Saving output:
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_dolpc.png"), img_DoLP_color)
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_dolpm.png"), img_DoLP_mono)
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_rgb.png"), img_rgb)
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_mono.png"), img_mono)
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_rgb90.png"), img_rgb_90)
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_aolpdolpm.png"), img_AoLP_DoLP_mono)
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_aolpdolpb.png"), img_AoLP_DoLP_color[..., 0])
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_aolpdolpg.png"), img_AoLP_DoLP_color[..., 1])
# cv2.imwrite(os.path.join(args.image_dir,img_token+"_aolpdolpr.png"), img_AoLP_DoLP_color[..., 2])
