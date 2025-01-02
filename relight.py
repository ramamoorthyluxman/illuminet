import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from tqdm import tqdm
from train import IllumiNet
from utils import params

def test(lps_cartesian, image_paths, model_path=None, image_size=(256, 256)):
    """
    Test the IllumiNet model and save reconstructed images.
    
    Args:
        lps_cartesian (np.ndarray): Light positions in Cartesian coordinates (N x 3)
        image_paths (list): List of paths where reconstructed images should be saved
        model_path (str, optional): Path to the model weights. If None, uses latest model
        image_size (tuple): Size of output images (height, width)
    """
    # Validate inputs
    if len(lps_cartesian) != len(image_paths):
        raise ValueError("Number of light positions must match number of image paths")
        
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = IllumiNet()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Load model weights
    if model_path is None:
        # Find the latest model in the saved_models directory
        model_dir = params.RTI_MODEL_SAVE_DIR
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist!")
            
        saved_models = [d for d in os.listdir(model_dir) 
                       if d.startswith('saved_models_')]
        if not saved_models:
            raise ValueError("No saved models found!")
            
        latest_model = max(saved_models)
        model_path = os.path.join(model_dir, latest_model, 'best_model.pth')
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist!")
        
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Convert light positions to tensor
    light_directions = torch.FloatTensor(lps_cartesian).to(device)

    # Create fixed-size input tensor
    # Create random normal input tensor instead of zeros

    sample_path = "/work/imvia/ra7916lu/illuminet/data/subset/painting1/LIGHT.JPG"
    sample_img = cv2.imread(sample_path)
    
    # Convert BGR to RGB and normalize to [0,1]
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    sample_img = sample_img.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    template_tensor = torch.FloatTensor(sample_img).permute(2, 0, 1).unsqueeze(0).to(device)
    

    # Process each light position
    print("Generating reconstructed images...")
    with torch.no_grad():
        for lp, save_path in tqdm(zip(light_directions, image_paths), 
                                total=len(image_paths), 
                                desc="Processing images"):
            try:
                # Prepare light direction input
                lp = lp.unsqueeze(0)  # Add batch dimension
                
                # Generate image
                output = model(template_tensor, lp)

                # Print values for debugging
                print(f"\nOutput tensor stats:")
                print(f"Shape: {output.shape}")
                print(f"Min: {output.min().item():.4f}")
                print(f"Max: {output.max().item():.4f}")
                print(f"Mean: {output.mean().item():.4f}")
                
                # Ensure output is in [0,1] range before converting to uint8
                output = torch.clamp(output, 0, 1)
                output_img = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save the image
                print("Writing image to: ", save_path)
                cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing image {save_path}: {str(e)}")
                continue

    print(f"Testing completed. Images saved to their respective paths.")
    return True

