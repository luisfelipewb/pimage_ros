import os, time
import cv2
import numpy as np
import polanalyser as pa
from pimage_lib import pimage as pimg


img_dir = "../img/"
img_file = "frame00000_raw.png"
img_token = "frame00000"
img_path = os.path.join(img_dir, img_file)

# Open image
img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print(f"Opening image {img_path} {img_raw.shape}")

################################################################################
# External library evaluation

# Track total time
print("\nExternal library (polanalyser)")
p_time = time.time()

# Extract demosaiced rgb images
s_time = time.time()
demosaiced_color = pa.demosaicing(img_raw, pa.COLOR_PolarRGB)
e_time = time.time()
print(f"Demosaic:\t {e_time-s_time:.4f} (s) -> {len(demosaiced_color)}x{demosaiced_color[0].shape} {demosaiced_color[0].dtype}")

# Extract monocolor polarization channels
s_time = time.time()
demosaiced_mono = []
for i in range(4):
    demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))
e_time = time.time()
print(f"Grayscale:\t {e_time-s_time:.4f} (s) -> {len(demosaiced_mono)}x{demosaiced_mono[0].shape} {demosaiced_mono[0].dtype}")

# Compute stokes parameters for monochrome
s_time = time.time()
radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
stokes_mono = pa.calcStokes(demosaiced_mono, radians)
e_time = time.time()
print(f"Stokes:   \t {e_time-s_time:.4f} (s) -> {stokes_mono.shape} {stokes_mono.dtype}")

# Compute DoLP values for monochrome
s_time = time.time()
val_DoLP_mono  = pa.cvtStokesToDoLP(stokes_mono) # 0~1
e_time = time.time()
print(f"DoLP Value:\t {e_time-s_time:.4f} (s) -> {val_DoLP_mono.shape} {val_DoLP_mono.dtype}")


# Compute AoLP values
s_time = time.time()
val_AoLP_mono = pa.cvtStokesToAoLP(stokes_mono)
e_time = time.time()
print(f"AoLP Value:\t {e_time-s_time:.4f} (s) -> {val_AoLP_mono.shape} {val_AoLP_mono.dtype}")


# Generate False colored AoLP_DoLP representation for monochrome
s_time = time.time()
img_AoLP_DoLP_mono = pa.applyColorToAoLP(val_AoLP_mono, saturation=1.0, value=val_DoLP_mono)
e_time = time.time()
print(f"False Color:\t {e_time-s_time:.4f} (s) -> {img_AoLP_DoLP_mono.shape} {img_AoLP_DoLP_mono.dtype}")

print(f"Total time:\t {time.time()-p_time:.4f}")

# Saving output:
cv2.imwrite(os.path.join(img_dir,img_token+"_aolpdolpm_pa.png"), img_AoLP_DoLP_mono)

################################################################################
# Own implementation evaluation

# Track total time
print("\nCustom implementation")
p_time = time.time()

# Extract demosaiced rgb images
s_time = time.time()
demosaiced_color = pimg.demosaicing(img_raw)
e_time = time.time()
print(f"Demosaic:\t {e_time-s_time:.4f} (s) -> {len(demosaiced_color)}x{demosaiced_color[0].shape} {demosaiced_color[0].dtype}")

# Extract monochorme polarization channels
s_time = time.time()
demosaiced_mono = []
for i in range(4):
    demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))
e_time = time.time()
print(f"Grayscale:\t {e_time-s_time:.4f} (s) -> {len(demosaiced_mono)}x{demosaiced_mono[0].shape} {demosaiced_mono[0].dtype}")


# Compute stokes parameters for monochrome
s_time = time.time()
stokes_mono = pimg.calcStokes(demosaiced_mono)
e_time = time.time()
print(f"Stokes:   \t {e_time-s_time:.4f} (s) -> {stokes_mono.shape} {stokes_mono.dtype}")

# Compute DoLP values for monochrome
s_time = time.time()
val_DoLP_mono  = pimg.calcDoLP(stokes_mono) # 0~1
e_time = time.time()
print(f"DoLP Value:\t {e_time-s_time:.4f} (s) -> {val_DoLP_mono.shape} {val_DoLP_mono.dtype}")

# Compute AoLP values
s_time = time.time()
val_AoLP_mono = pimg.calcAoLP(stokes_mono)
e_time = time.time()
print(f"AoLP Value:\t {e_time-s_time:.4f} (s) -> {val_AoLP_mono.shape} {val_AoLP_mono.dtype}")

# Generate False colored AoLP_DoLP representation for monochrome
s_time = time.time()
img_AoLP_DoLP_mono = pimg.falseColoring(val_AoLP_mono, val_DoLP_mono)
e_time = time.time()
print(f"False Color:\t {e_time-s_time:.4f} (s) -> {img_AoLP_DoLP_mono.shape} {img_AoLP_DoLP_mono.dtype}")

print(f"Total time:\t {time.time()-p_time:.4f}")

# Saving output:
cv2.imwrite(os.path.join(img_dir,img_token+"_aolpdolpm_pi.png"), img_AoLP_DoLP_mono)