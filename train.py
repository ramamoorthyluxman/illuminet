import os
import sys
import math
from datetime import datetime
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import params

class TruePixelwiseMLPRelight(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[8, 8, 8], patch_size=8):
        super().__init__()
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.num_patches_h = 1024 // patch_size  # Should be 128 for patch_size=8
        self.num_patches_w = 1024 // patch_size  # Should be 128 for patch_size=8
        
        print(f"Number of patches: {self.num_patches_h}x{self.num_patches_w}")
        
        # For each patch, create a set of weights (sharing within patch)
        # Layer 1: input_dim -> hidden_dims[0]
        self.weights1 = nn.Parameter(torch.randn(self.num_patches_h, self.num_patches_w, 
                                               hidden_dims[0], input_dim) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(self.num_patches_h, self.num_patches_w, 
                                            hidden_dims[0]))
        
        # Hidden layers
        self.weights2 = nn.Parameter(torch.randn(self.num_patches_h, self.num_patches_w, 
                                               hidden_dims[1], hidden_dims[0]) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(self.num_patches_h, self.num_patches_w, 
                                            hidden_dims[1]))
        
        self.weights3 = nn.Parameter(torch.randn(self.num_patches_h, self.num_patches_w, 
                                               hidden_dims[2], hidden_dims[1]) * 0.01)
        self.bias3 = nn.Parameter(torch.zeros(self.num_patches_h, self.num_patches_w, 
                                            hidden_dims[2]))
        
        # Output layer: hidden_dims[-1] -> 3 (RGB)
        self.weights_out = nn.Parameter(torch.randn(self.num_patches_h, self.num_patches_w, 
                                                  3, hidden_dims[-1]) * 0.01)
        self.bias_out = nn.Parameter(torch.zeros(self.num_patches_h, self.num_patches_w, 3))

    def forward(self, light_direction):
        batch_size = light_direction.shape[0]
        
        # Expand light direction for patches
        # [batch_size, 2] -> [batch_size, num_patches_h, num_patches_w, 2]
        x = light_direction.unsqueeze(1).unsqueeze(1).expand(
            -1, self.num_patches_h, self.num_patches_w, -1)
        
        # Layer 1
        hidden = torch.sum(x.unsqueeze(-2) * self.weights1, dim=-1) + self.bias1
        hidden = F.relu(hidden)
        
        # Layer 2
        hidden = torch.sum(hidden.unsqueeze(-2) * self.weights2, dim=-1) + self.bias2
        hidden = F.relu(hidden)
        
        # Layer 3
        hidden = torch.sum(hidden.unsqueeze(-2) * self.weights3, dim=-1) + self.bias3
        hidden = F.relu(hidden)
        
        # Output layer
        output = torch.sum(hidden.unsqueeze(-2) * self.weights_out, dim=-1) + self.bias_out
        output = torch.sigmoid(output)  # [batch_size, 128, 128, 3]
        
        # Reshape output to match image size by repeating patch values
        # First, reshape to [batch_size, 128, 1, 128, 1, 3]
        output = output.unsqueeze(2).unsqueeze(4)
        
        # Repeat patch values
        output = output.repeat(1, 1, self.patch_size, 1, self.patch_size, 1)
        
        # Finally, reshape to image size [batch_size, 1024, 1024, 3]
        output = output.reshape(batch_size, 1024, 1024, 3)
        
        return output
    
class PixelwiseDataset(Dataset):
    def __init__(self, light_directions, target_images):
        """
        Args:
            light_directions: (N, 2) - Light positions (azimuth, elevation)
            target_images: (N, H, W, 3) - Target RGB images
        """
        self.light_directions = torch.FloatTensor(light_directions)
        # Convert target images to float and normalize to [0, 1]
        self.target_images = torch.FloatTensor(target_images) / 255.0
        
        print(f"Target images min: {self.target_images.min()}, max: {self.target_images.max()}")
        
        # Normalize light directions
        self.light_directions = (self.light_directions - self.light_directions.mean(0)) / self.light_directions.std(0)
        print(f"Light directions min: {self.light_directions.min()}, max: {self.light_directions.max()}")

    def __len__(self):
        return len(self.light_directions)

    def __getitem__(self, idx):
        return {
            'light_direction': self.light_directions[idx],
            'target_image': self.target_images[idx]
        }

def save_comparison_images(model, val_loader, device, epoch, model_save_path, num_samples=4):
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        light_directions = batch['light_direction'][:num_samples].to(device)
        target_images = batch['target_image'][:num_samples].to(device)
        
        outputs = model(light_directions)
        
        # Print ranges before visualization
        print("\nVisualization Stats:")
        print(f"Target range: min={target_images.min().item():.4f}, max={target_images.max().item():.4f}")
        print(f"Output range: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
        for i in range(num_samples):
            axes[i, 0].imshow(target_images[i].cpu().numpy())
            axes[i, 0].set_title('Target')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(outputs[i].cpu().numpy())
            axes[i, 1].set_title('Output')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, f'comparison_epoch_{epoch}.png'))
        plt.close()

def plot_loss_evolution(train_losses, val_losses, model_save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training MSE')
    plt.plot(val_losses, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Validation Error Evolution')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(os.path.join(model_save_path, 'loss_evolution.png'))
    plt.close()

def train(light_directions, target_images):
    print("Starting training process...")
    print(f"Input target_images range: min={np.min(target_images)}, max={np.max(target_images)}")
    
    # Create save directory
    model_save_path = os.path.join(params.RTI_MODEL_SAVE_DIR, f"saved_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Split data
    train_light_dirs, val_light_dirs, train_targets, val_targets = train_test_split(
        light_directions, target_images, test_size=params.TRAIN_VAL_SPLIT, random_state=42
    )

    # Create datasets and loaders
    train_dataset = PixelwiseDataset(train_light_dirs, train_targets)
    val_dataset = PixelwiseDataset(val_light_dirs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model with smaller network and patch-based approach
    model = TruePixelwiseMLPRelight(
        input_dim=2,
        hidden_dims=[8, 8, 8],  # Much smaller network
        patch_size=8  # Share weights within 8x8 patches
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # Use AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=params.LEARNING_RATE, weight_decay=0.01)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                   patience=5, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    patience = 15  # Increased patience
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(params.RTI_NET_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            light_directions = batch['light_direction'].to(device)
            target_images = batch['target_image'].to(device)
            
            optimizer.zero_grad()
            outputs = model(light_directions)
            loss = criterion(outputs, target_images)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx == 0:
                print(f"\nBatch Stats (Epoch {epoch+1}):")
                print(f"Target range: min={target_images.min().item():.4f}, max={target_images.max().item():.4f}")
                print(f"Output range: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
                print(f"Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                light_directions = batch['light_direction'].to(device)
                target_images = batch['target_image'].to(device)
                
                outputs = model(light_directions)
                loss = criterion(outputs, target_images)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{params.RTI_NET_EPOCHS}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Save periodic checkpoints
        if (epoch + 1) % params.RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS == 0:
            save_comparison_images(model, val_loader, device, epoch + 1, model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth'))
            plot_loss_evolution(train_losses, val_losses, model_save_path)

    print("Training completed successfully.")
    return model