import os
import random
import numpy as np

def get_image_files(directory):
    """Get a list of .nii.gz image files in the given directory, excluding _label.nii.gz files."""
    return [f for f in os.listdir(directory) if (f.endswith('.nii.gz') and not f.endswith('_labels.nii.gz'))]

def create_image_pairs(image_files):
    """Create random pairs of images from the list."""
    pairs = []
    random.shuffle(image_files)
    while len(image_files) > 1:
        fixed_image = image_files.pop()
        moving_image = image_files.pop()
        pairs.append((fixed_image, moving_image))
    return pairs

def save_pairs_to_npy(pairs, file_path):
    """Save the image pairs to a .npy file."""
    np.save(file_path, pairs)

def load_pairs_from_npy(file_path):
    """Load the image pairs from a .npy file."""
    pairs = np.load(file_path, allow_pickle=True)
    return pairs.tolist()

# Directory containing the training images
directory = '/home/vrdoc/GF/lung_registration/data/OASIS3_Dataset/validation/'

# Get list of image files
image_files = get_image_files(directory)

# Create random pairs of images
image_pairs = create_image_pairs(image_files)

# File path to save the .npy file
npy_file_path = os.path.join(directory, 'image_pairs.npy')

# Save pairs to .npy file
save_pairs_to_npy(image_pairs, npy_file_path)

# Load pairs from .npy file
loaded_pairs = load_pairs_from_npy(npy_file_path)

# Print the loaded pairs
print(loaded_pairs)