import cv2
import os

# Function to convert the image from RGB to YCbCr
def rgb_to_ycbcr(image):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return ycbcr_image

# Load dataset folder and get the image names
dataset_folder = 'dataset/images'
image_names = os.listdir(dataset_folder)

# Create Results directory if it doesn't exist
results_dir = 'Results'
os.makedirs(results_dir, exist_ok=True)

# Loop through the images in the dataset folder
for image_name in image_names:  # Process first 5 images for testing
    # Load the image
    image_path = os.path.join(dataset_folder, image_name)
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Convert the image from RGB to YCbCr
    ycbcr_image = rgb_to_ycbcr(rgb_image)

    #split the YCbCr image into Y, Cb, and Cr channels
    Y, Cb, Cr = cv2.split(ycbcr_image)

    # Save the YCbCr image
    ycbcr_image_path = os.path.join(results_dir, 'YCbCr_' + image_name)
    cv2.imwrite(ycbcr_image_path, Y)

    # show progress to the user as percentage
    print('YCbCr image saved for', image_name)


print('YCbCr images saved in the Results directory')
    

