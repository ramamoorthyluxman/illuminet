import os
import shutil
import random
import glob

def split_rti_dataset(acq_path, split_save_path, test_percentage):
    """
    Split RTI acquisition data into training and test sets.
    
    Args:
        acq_path (str): Path to the RTI acquisition folder
        test_percentage (float): Percentage of data to use for test (0-100)
    """
    # Convert percentage to decimal
    test_ratio = test_percentage / 100.0
    
    # Check if the path exists
    if not os.path.exists(acq_path):
        raise ValueError(f"Acquisition path {acq_path} does not exist")
    
    # Find .lp file
    lp_files = glob.glob(os.path.join(acq_path, "*.lp"))
    if not lp_files:
        raise ValueError(f"No .lp file found in {acq_path}")
    lp_file = lp_files[0]  # Take the first .lp file
    
    # Read the .lp file
    with open(lp_file, 'r') as f:
        lp_content = f.readlines()
    
    # Parse the header line (number of images)
    try:
        num_images = int(lp_content[0].strip())
        print(f"Number of images in acquisition: {num_images}")
    except (IndexError, ValueError):
        raise ValueError("Invalid .lp file format: First line should contain number of images")
    
    # Extract image filenames and light positions (starting from second line)
    image_data = []  # List to store tuples of (image_name, light_position)
    for line in lp_content[1:]:  # Skip the first line
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            if len(parts) >= 4:  # Image name + 3 coordinates
                image_name = parts[0]
                light_position = ' '.join(parts[1:4])  # Keep light position data
                image_data.append((image_name, light_position))
    
    # Verify number of images matches header
    if len(image_data) != num_images:
        print(f"Warning: Number of images in header ({num_images}) doesn't match actual number of images ({len(image_data)})")
    
    # Randomly shuffle the image data
    random.seed(42)  # For reproducibility
    random.shuffle(image_data)
    
    # Calculate split indices
    test_size = int(len(image_data) * test_ratio)
    test_data = image_data[:test_size]
    train_data = image_data[test_size:]
    
    # Create train and test directories
    train_dir = os.path.join(split_save_path, 'train')
    test_dir = os.path.join(split_save_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    def create_lp_file(output_path, data_subset):
        """Create new .lp file with subset of images, preserving light position data"""
        # Write header with number of images in subset
        new_lp_content = [f"{len(data_subset)}\n"]
        
        # Add image data with light positions
        for image_name, light_position in data_subset:
            new_lp_content.append(f"{image_name} {light_position}\n")
        
        with open(os.path.join(output_path, os.path.basename(lp_file)), 'w') as f:
            f.writelines(new_lp_content)
    
    # Copy images and create new .lp files
    def copy_images(data_subset, dest_dir):
        for image_name, _ in data_subset:
            src_path = os.path.join(acq_path, image_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_dir)
            else:
                print(f"Warning: Image {image_name} not found")
    
    # Process training set
    copy_images(train_data, train_dir)
    create_lp_file(train_dir, train_data)
    
    # Process test set
    copy_images(test_data, test_dir)
    create_lp_file(test_dir, test_data)
    
    print(f"Dataset split complete:")
    print(f"Training set: {len(train_data)} images")
    print(f"test set: {len(test_data)} images")

# Example usage
if __name__ == "__main__":
    # Replace with your acquisition path and desired test percentage
    acq_path = r"/work/imvia/ra7916lu/illuminet/data/subset/painting1/dense"
    split_save_path = r"/work/imvia/ra7916lu/illuminet/data/subset/painting1/sparse" 
    test_percentage = 80  # 20% for test
    split_rti_dataset(acq_path, split_save_path, test_percentage)