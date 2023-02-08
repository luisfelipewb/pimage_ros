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
s_time = time.time()
demosaiced_color = pi.demosaicing(img_raw)
e_time = time.time()
print(f"Demosaic:\t {e_time-s_time:.4f} (s) -> {len(demosaiced_color)}x{demosaiced_color[0].shape} {demosaiced_color[0].dtype}")

# Extract monochorme polarization channels
s_time = time.time()
demosaiced_mono = []
for i in range(4):
    demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))
e_time = time.time()
print(f"Grayscale:\t {e_time-s_time:.4f} (s) -> {len(demosaiced_mono)}x{demosaiced_mono[0].shape} {demosaiced_mono[0].dtype}")

# Extract regular RGB image (I_0 + I_90)
# img_rgb = pi.rgb(demosaiced_color)

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
s_time = time.time()
stokes_mono = pi.calcStokes(demosaiced_mono)
e_time = time.time()
print(f"Stokes:   \t {e_time-s_time:.4f} (s) -> {stokes_mono.shape} {stokes_mono.dtype}")

# Compute DoLP values for each color: (2048, 2448, 3) float64
# val_DoLP_color  = pa.cvtStokesToDoLP(stokes_color) # 0~1
# img_DoLP_color = (val_DoLP_color * 255).round().astype(np.uint8)

# Compute DoLP values for monochrome: (2048, 2448, 1) float64
s_time = time.time()
val_DoLP_mono  = pi.calcDoLP(stokes_mono) # 0~1
e_time = time.time()
print(f"DoLP Value:\t {e_time-s_time:.4f} (s) -> {val_DoLP_mono.shape} {val_DoLP_mono.dtype}")

s_time = time.time()
img_DoLP_mono = (val_DoLP_mono * 255).round().astype(np.uint8)
e_time = time.time()
print(f"DoLP Color:\t {e_time-s_time:.4f} (s) -> {img_DoLP_mono.shape} {img_DoLP_mono.dtype}")

# Compute AoLP values: (2048, 2448, 3) float64
# val_AoLP_color = pa.cvtStokesToAoLP(stokes_color)

# Compute AoLP values: (2048, 2448, 1) float64
s_time = time.time()
val_AoLP_mono = pi.calcAoLP(stokes_mono)
e_time = time.time()
print(f"AoLP Value:\t {e_time-s_time:.4f} (s) -> {val_AoLP_mono.shape} {val_AoLP_mono.dtype}")

# Generate False colored AoLP_DoLP representation for all three color channels
# img_AoLP_DoLP_color = np.empty(stokes_color.shape, dtype=img_DoLP_color.dtype)
# for i in range(3):
    # img_AoLP_DoLP_color[..., i] = pa.applyColorToAoLP(val_AoLP_color[:,:,i], saturation=1.0, value=val_DoLP_color[:,:,i])

# Generate False colored AoLP_DoLP representation for monochrome
s_time = time.time()
img_AoLP_DoLP_mono = pi.falseColoring(val_AoLP_mono, value=val_DoLP_mono)
e_time = time.time()
print(f"False Color:\t {e_time-s_time:.4f} (s) -> {img_AoLP_DoLP_mono.shape} {img_AoLP_DoLP_mono.dtype}")

print(f"Total time:\t {time.time()-p_time:.4f}")


# Saving output:
# cv2.imwrite(os.path.join(img_dir,img_token+"_dolpc_pi.png"), img_DoLP_color)
cv2.imwrite(os.path.join(img_dir,img_token+"_dolpm_pi.png"), img_DoLP_mono)
# cv2.imwrite(os.path.join(img_dir,img_token+"_rgb_pi.png"), img_rgb)
# cv2.imwrite(os.path.join(img_dir,img_token+"_mono_pi.png"), img_mono)
# cv2.imwrite(os.path.join(img_dir,img_token+"_rgb90_pi.png"), img_rgb_90)
cv2.imwrite(os.path.join(img_dir,img_token+"_aolpdolpm_pi.png"), img_AoLP_DoLP_mono)
# cv2.imwrite(os.path.join(img_dir,img_token+"_aolpdolpb_pi.png"), img_AoLP_DoLP_color[..., 0])
# cv2.imwrite(os.path.join(img_dir,img_token+"_aolpdolpg_pi.png"), img_AoLP_DoLP_color[..., 1])
# cv2.imwrite(os.path.join(img_dir,img_token+"_aolpdolpr_pi.png"), img_AoLP_DoLP_color[..., 2])
