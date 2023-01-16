import argparse, os, time
import cv2
import numpy as np
import polanalyser as pa

parser = argparse.ArgumentParser(description="Test pipeline using polanalyser.")
parser.add_argument("image_dir", help="Directory where at least one raw image exists")
args = parser.parse_args()

if os.path.isdir(args.image_dir) is False:
    print("Directory", args.image_dir,"does not exists")
    quit()

img_name = None
img_token = None
for file in os.listdir(args.image_dir):
    if "_raw.png" in file:
        img_name = file
        img_token = img_name[:-8] # remove '_raw.png' ending
        break

if img_name is None:
    print("raw image not found in", args.image_dir)
    quit()

print(img_name)

# Open image
s_time = time.time()
img_raw = cv2.imread("/home/lwolfbat/bags/sample/images/frame00000_raw.png", cv2.IMREAD_GRAYSCALE)
e_time = time.time()
print("read time:\t", e_time - s_time)

# Track total time
p_time = time.time()

# Extract demosaiced rgb images: 4x(2048, 2448, 3) uint8
s_time = time.time()
demosaiced_color = pa.demosaicing(img_raw, pa.COLOR_PolarRGB)
e_time = time.time()
print("demosaicing_c:\t", e_time - s_time, "->", len(demosaiced_color), "x", demosaiced_color[0].shape, demosaiced_color[0].dtype)

# Extract monocolor polarization channels
s_time = time.time()
demosaiced_mono = []
for i in range(4):
    demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))
e_time = time.time()
print("demosaicing_m:\t", e_time - s_time, "->",  demosaiced_mono[0].shape, demosaiced_mono[0].dtype)

# Extract regular RGB image (I_0 + I_90)
s_time = time.time()
img_rgb = np.empty((2048, 2448, 3), demosaiced_color[0].dtype)
for i in range(3):
    img_0 = demosaiced_color[0][...,i] 
    img_90 = demosaiced_color[2][...,i]
    img_rgb[...,i] = cv2.addWeighted(img_0, 0.5, img_90, 0.5, 0.0)
e_time = time.time()
print("regular rgb:\t", e_time - s_time, "->",  img_rgb.shape, img_rgb.dtype)

# Same as regular filter
img_rgb_90 = demosaiced_color[2]

# Monochrome
s_time = time.time()
img_mono = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
e_time = time.time()
print("regular_mono:\t", e_time - s_time, "->",  img_mono.shape, img_mono.dtype)

# Compute stokes parameters for each color: (2048, 2448, 3, 3) float64
s_time = time.time()
radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
stokes_color = pa.calcStokes(demosaiced_color, radians)
e_time = time.time()
print("stokes (c):\t", e_time - s_time, "->", stokes_color.shape, stokes_color.dtype)

# Compute stokes parameters for monochrome: (2048, 2448, 3) float64
s_time = time.time()
radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
stokes_mono = pa.calcStokes(demosaiced_mono, radians)
e_time = time.time()
print("stokes (m):\t", e_time - s_time, "->", stokes_mono.shape, stokes_mono.dtype)

# Compute DoLP values for each color: (2048, 2448, 3) float64
s_time = time.time()
val_DoLP_color  = pa.cvtStokesToDoLP(stokes_color) # 0~1
e_time = time.time()
print("dolp value:\t", e_time - s_time, "->", val_DoLP_color.shape, val_DoLP_color.dtype)

# Compute DoLP values for monochrome: (2048, 2448, 3) float64
s_time = time.time()
val_DoLP_mono  = pa.cvtStokesToDoLP(stokes_mono) # 0~1
e_time = time.time()
print("dolp value:\t", e_time - s_time, "->", val_DoLP_mono.shape, val_DoLP_mono.dtype)

# Generate DoLP image: (2048, 2448, 3) uint8
s_time = time.time()
img_DoLP_color = (val_DoLP_color * 255).round().astype(np.uint8)
e_time = time.time()
print("dolp image (c):\t", e_time - s_time, "->", img_DoLP_color.shape, img_DoLP_color.dtype)

s_time = time.time()
img_DoLP_mono = (val_DoLP_mono * 255).round().astype(np.uint8)
e_time = time.time()
print("dolp image (m):\t", e_time - s_time, "->", img_DoLP_mono.shape, img_DoLP_mono.dtype)

# Compute AoLP values: (2048, 2448, 3) float64
s_time = time.time()
val_AoLP_color = pa.cvtStokesToAoLP(stokes_color)
e_time = time.time()
print("alp val:\t", e_time - s_time, "->", val_AoLP_color.shape, val_AoLP_color.dtype)

# Compute AoLP values: (2048, 2448, 3) float64
s_time = time.time()
val_AoLP_mono = pa.cvtStokesToAoLP(stokes_mono)
e_time = time.time()
print("alp val:\t", e_time - s_time, "->", val_AoLP_mono.shape, val_AoLP_mono.dtype)


# Generate False colored AoLP_DoLP representation for all three color channels
s_time = time.time()
img_AoLP_DoLP_color = np.empty(stokes_color.shape, dtype=img_DoLP_color.dtype)
for i in range(3):
    img_AoLP_DoLP_color[..., i] = pa.applyColorToAoLP(val_AoLP_color[:,:,i], saturation=1.0, value=val_DoLP_color[:,:,i])
e_time = time.time()
print("img_AoLP_DoLP\t", e_time - s_time, "->", img_AoLP_DoLP_color.shape, img_AoLP_DoLP_color.dtype)

# Generate False colored AoLP_DoLP representation for monochrome
s_time = time.time()
img_AoLP_DoLP_mono = pa.applyColorToAoLP(val_AoLP_mono, saturation=1.0, value=val_DoLP_mono)
e_time = time.time()
print("img_AoLP_DoLP_m\t", e_time - s_time, "->", img_AoLP_DoLP_color.shape, img_AoLP_DoLP_color.dtype)


print("Total time:\t", time.time() - p_time)


# Saving output:

base_name = "/home/lwolfbat/bags/sample/images/frame00000"
cv2.imwrite(os.path.join(args.image_dir,img_token+"_dolpc.png"), img_DoLP_color)
cv2.imwrite(os.path.join(args.image_dir,img_token+"_dolpm.png"), img_DoLP_mono)
cv2.imwrite(os.path.join(args.image_dir,img_token+"_rgb.png"), img_rgb)
cv2.imwrite(os.path.join(args.image_dir,img_token+"_mono.png"), img_mono)
cv2.imwrite(os.path.join(args.image_dir,img_token+"_rgb90.png"), img_rgb_90)
cv2.imwrite(os.path.join(args.image_dir,img_token+"_aolpdolpm.png"), img_AoLP_DoLP_mono)
cv2.imwrite(os.path.join(args.image_dir,img_token+"_aolpdolpb.png"), img_AoLP_DoLP_color[..., 0])
cv2.imwrite(os.path.join(args.image_dir,img_token+"_aolpdolpg.png"), img_AoLP_DoLP_color[..., 1])
cv2.imwrite(os.path.join(args.image_dir,img_token+"_aolpdolpr.png"), img_AoLP_DoLP_color[..., 2])
