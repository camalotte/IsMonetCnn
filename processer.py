from PIL import Image
import os

# Define the directory where the images are stored and the size to which we want to resize them
image_directory = 'monet_images'
target_size = (256, 256)  # You can change this to whatever size you need


# Function to process and resize images
def process_and_resize_images(image_directory, target_size):
    # List all .jpg files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]

    for image_file in image_files:
        # Open the image
        with Image.open(os.path.join(image_directory, image_file)) as img:
            # Convert the image to RGB (in case it's a different mode)
            img = img.convert('RGB')
            # Resize the image using the LANCZOS filter
            img_resized = img.resize(target_size, Image.LANCZOS)
            # Save the processed image to a new directory
            img_resized.save(os.path.join('processed_images', f'resized_{image_file}'), 'JPEG', quality=90)


# Call the function to process the images
process_and_resize_images(image_directory, target_size)
