import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rgb_to_ycbcr(image):
    # Convert the image from RGB to YCbCr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return ycbcr_image

# Load an example image (replace 'example.jpg' with your image path)
image_path = 'Example.png'
rgb_image = cv2.imread(image_path)

# Convert the image from BGR (OpenCV default) to RGB
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(rgb_image)

r_size = np.size(r)
g_size = np.size(g)
b_size = np.size(b)

print(f"size of R component {r_size}")
print(f"size of G component {g_size}")
print(f"size of B component {b_size}")

# Convert RGB image to YCbCr
ycbcr_image = rgb_to_ycbcr(rgb_image)

# Split the channels
Y, Cb, Cr = cv2.split(ycbcr_image)

Y_size = np.size(Y)
Cb_size = np.size(Cb)
Cr_size = np.size(Cr)

print(f"size of Y component {Y_size}")
print(f"size of Cb component {Cb_size}")
print(f"size of Cr component {Cr_size}")

# Display the images
plt.figure(figsize=(12, 8))
plt.subplot(1, 4, 1)
plt.imshow(rgb_image)
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(Y, cmap='gray')
plt.title('Y Channel')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(Cb, cmap='gray')
plt.title('Cb Channel')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(Cr, cmap='gray')
plt.title('Cr Channel')
plt.axis('off')

plt.show()

# Plot RGB datapoints in 3D
fig = plt.figure(figsize=(14, 6))

ax = fig.add_subplot(121, projection='3d')
r, g, b = rgb_image[..., 0].flatten(), rgb_image[..., 1].flatten(), rgb_image[..., 2].flatten()
ax.scatter(r, g, b, c=rgb_image.reshape(-1, 3)/255.0, marker='o')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_title('RGB Data Points')

# Plot YCbCr datapoints in 3D
ax = fig.add_subplot(122, projection='3d')
y, cb, cr = Y.flatten(), Cb.flatten(), Cr.flatten()
ax.scatter(y, cb, cr, c=rgb_image.reshape(-1, 3)/255.0, marker='o')
ax.set_xlabel('Y')
ax.set_ylabel('Cb')
ax.set_zlabel('Cr')
ax.set_title('YCbCr Data Points')

plt.show()

# Plot Y component in 1D
plt.figure(figsize=(10, 4))
plt.plot(Y.flatten(), color='gray', lw=0.5)
plt.title('Y Component')
plt.xlabel('Pixel Index')
plt.ylabel('Y Value')
plt.grid(True)
plt.show()
