import os, time
import cv2
import numpy as np
from pimage_lib import pimage as pi

img_dir = "../img/"
img_file = "frame00000_raw.png"
img_token = "frame00000"
img_path = os.path.join(img_dir, img_file)
print(f"Opening image {img_path}")

# Open image
img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Track total time
p_time = time.time()

# Extract demosaiced rgb images: 4x(2048, 2448, 3) uint8
demosaiced_color = pi.demosaicing(img_raw)

# Extract monochorme polarization channels
demosaiced_mono = []
for i in range(4):
    demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))

# Extract regular RGB image (I_0 + I_90)
img_rgb = pi.rgb(demosaiced_color)

# Same as regular filter
img_rgb_90 = demosaiced_color[2]

# Monochrome
img_mono = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Compute stokes parameters for each color: (2048, 2448, 3, 3) float64
stokes_color = pi.calcStokes(demosaiced_color)
print(stokes_color.shape)

# Compute stokes parameters for monochrome: (2048, 2448, 3) float64
stokes_mono = pi.calcStokes(demosaiced_mono)

# Compute DoLP values for each color: (2048, 2448, 3) float64
val_DoLP_color  = pi.calcDoLP(stokes_color) # 0~1

# Compute DoLP values for monochrome: (2048, 2448, 1) float64
val_DoLP_mono  = pi.calcDoLP(stokes_mono) # 0~1

img_DoLP_mono = (val_DoLP_mono * 255).round().astype(np.uint8)
img_DoLP_color = (val_DoLP_color * 255).round().astype(np.uint8)

# Compute AoLP values: (2048, 2448, 3) float64
val_AoLP_color = pi.calcAoLP(stokes_color)

# Compute AoLP values: (2048, 2448, 1) float64
val_AoLP_mono = pi.calcAoLP(stokes_mono)

# Generate False colored AoLP_DoLP representation for all three color channels
img_AoLP_DoLP_color = np.empty(stokes_color.shape, dtype=img_DoLP_color.dtype)
for i in range(3):
    img_AoLP_DoLP_color[..., i] = pi.falseColoring(val_AoLP_color[...,i], value=val_DoLP_color[...,i])

# Generate False colored AoLP_DoLP representation for monochrome
img_AoLP_DoLP_mono = pi.falseColoring(val_AoLP_mono, value=val_DoLP_mono)

img_rgb_dif = pi.calcDiffuse(stokes_color)
img_rgb_spc = pi.calcDoLP(stokes_color)
print(f"Total time:\t {time.time()-p_time:.4f}")


# Saving output:
cv2.imwrite(os.path.join(img_dir,img_token+"_dolpc_pi.png"), img_DoLP_color)
cv2.imwrite(os.path.join(img_dir,img_token+"_dolp_pi.png"), img_DoLP_mono)
cv2.imwrite(os.path.join(img_dir,img_token+"_rgb_pi.png"), img_rgb)
cv2.imwrite(os.path.join(img_dir,img_token+"_mono_pi.png"), img_mono)
cv2.imwrite(os.path.join(img_dir,img_token+"_rgb90_pi.png"), img_rgb_90)
cv2.imwrite(os.path.join(img_dir,img_token+"_pol_pi.png"), img_AoLP_DoLP_mono)
cv2.imwrite(os.path.join(img_dir,img_token+"_polb_pi.png"), img_AoLP_DoLP_color[..., 0])
cv2.imwrite(os.path.join(img_dir,img_token+"_polg_pi.png"), img_AoLP_DoLP_color[..., 1])
cv2.imwrite(os.path.join(img_dir,img_token+"_polr_pi.png"), img_AoLP_DoLP_color[..., 2])
cv2.imwrite(os.path.join(img_dir,img_token+"_rgbspc_pi.png"), img_rgb_dif)
cv2.imwrite(os.path.join(img_dir,img_token+"_rgbdif_pi.png"), img_rgb_spc)
