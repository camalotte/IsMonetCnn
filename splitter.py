import os
import shutil
from sklearn.model_selection import train_test_split

# Directories for the dataset
base_dir = 'processed_images'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Create directories in the root folder if they don't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)


# Get all image filenames
image_files = [f for f in os.listdir(base_dir) if f.endswith('.jpg')]
image_paths = [os.path.join(base_dir, f) for f in image_files]

# Split the data into train and test sets (80% train, 20% test)
train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Split the train set further into training and validation sets (80% train, 20% validation)
train_paths, val_paths = train_test_split(train_paths, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Function to copy files to a target directory
def copy_files(file_list, target_dir):
    for file_path in file_list:
        shutil.copy(file_path, target_dir)

# Copy the images to the respective directories
copy_files(train_paths, train_dir)
copy_files(val_paths, val_dir)
copy_files(test_paths, test_dir)

print(f"Number of images in training set: {len(os.listdir(train_dir))}")
print(f"Number of images in validation set: {len(os.listdir(val_dir))}")
print(f"Number of images in test set: {len(os.listdir(test_dir))}")