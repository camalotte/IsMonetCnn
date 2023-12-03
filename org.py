import os
import shutil

# Base directory for split datasets
base_dir = 'split_data'

# Names of the new class subdirectories
monet_dir_name = 'Monet'
not_monet_dir_name = 'not_Monet'


# Function to create class subdirectories and move images
def create_class_subdirectories(base_dir, monet_dir_name, not_monet_dir_name):
    # Define the subdirectories for the dataset categories
    categories = ['train', 'val', 'test']

    for category in categories:
        # Create new subdirectories for the Monet and not_Monet classes
        monet_dir = os.path.join(base_dir, category, monet_dir_name)
        not_monet_dir = os.path.join(base_dir, category, not_monet_dir_name)
        os.makedirs(monet_dir, exist_ok=True)
        os.makedirs(not_monet_dir, exist_ok=True)

        # Move all images into the Monet subdirectory (since you only have Monet images right now)
        # Assuming all images are Monet's, if you have not_Monet images, you'll need to sort them accordingly
        for image_file in os.listdir(os.path.join(base_dir, category)):
            if image_file.endswith('.jpg'):
                # Construct the full file paths
                current_file_path = os.path.join(base_dir, category, image_file)
                new_file_path = os.path.join(monet_dir, image_file)

                # Move the file
                shutil.move(current_file_path, new_file_path)


# Run the function to adjust the directory structure
create_class_subdirectories(base_dir, monet_dir_name, not_monet_dir_name)
