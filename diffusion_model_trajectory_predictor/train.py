"""
Author: Giovanni Lucente, Marko Mizdrak

This script trains a model using different modalities (standard supervised learning or diffusion) 
to predict the trajectories of traffic participants for one second. The predicted trajectories are 
encoded in an image, where they are displayed as colored lines on a black background. 
The model receives the image of the trajectories from the previous second as input (or a conditioning image).

"""

import numpy as np
from loss_functions import *
from trajectory_dataset import *
from models.CNNPredictor import CNNPrediction
from models.Model1 import Model1
from models.ModelAttention import ModelAttention
from models.ConditionalVAE import ConditionalVAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import segmentation_models_pytorch as smp
import pytorch_ssim
import shutil
import types

# Configuration Parameters
IMAGE_DIR = 'output_images_cv2'  # Directory containing the images
OUTPUT_DIR = 'output'            # Directory to save the trained models and plots
TIME_STEPS = 1000                # number of steps in the noising and denoising process
LEARNING_RATE = 1e-4
DATASET_FRACTION = 1.0           # = 1 if using the whole dataset
VALIDATION_SPLIT = 0.2
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for image prediction models.')
    
    # Model selection
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unet_diff'],
                        help='Model type to use: "unet" or "unet_diff"')

    # Loss function selection
    parser.add_argument('--loss', type=str, default='weighted_l1', choices=['mse', 'weighted_mse', 'weighted_l1', 'ssim', 'dice', 'weighted_dice', 'weighted_ssim', 'perceptual', 'edge', 'color', 'wavelet'],
                        help='Loss function to use: "mse" (Mean Squared Error), "weighted_mse" (Weighted Mean Squared Error), "ssim" (SSIM), "dice" (Dice Loss), "weighted_dice", "weighted_ssim", "perceptual", "edge", "color", or "wavelet"')

    # Batch size selection
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    
    # Number of epochs
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')

    args = parser.parse_args()
    return args

def extract(a, t, x_shape):
    """Extract values at index t from tensor a and reshape to match x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out

# diffusion training loop
def train_diffusion_unet(model, diffusion, train_loader, val_loader, optimizer, epochs, device, output_dir, loss_fn):
    model = model.to(device)
    diffusion = diffusion.to(device)

    # Get model and loss function names
    model_name = type(model).__name__  # Get the class name of the model
    loss_fn_name = loss_fn.__name__ if isinstance(loss_fn, types.FunctionType) else loss_fn.__class__.__name__  # Get the name of the loss function
    
    # Create directory for saving the model
    model_save_dir = os.path.join(output_dir, f"{model_name}_diffusion_{loss_fn_name}")
    os.makedirs(model_save_dir, exist_ok=True)

    # Path to best model checkpoint
    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    best_val_loss = float("inf")  # Persist best loss across epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0 

        for condition, target in train_loader:
            # Move tensors to the GPU (if available)
            condition = condition.to(device)
            target = target.to(device)

            # Sample a random timestep
            t = torch.randint(0, diffusion.num_timesteps, (target.shape[0],), device=target.device)

            # Perform forward diffusion to create noisy image
            noisy_target = diffusion.q_sample(target, t)
            true_noise = (noisy_target - extract(diffusion.sqrt_alphas_cumprod, t, target.shape) * target) / extract(diffusion.sqrt_one_minus_alphas_cumprod, t, target.shape)

            # Concatenate noisy target with conditioning
            model_input = torch.cat([noisy_target, condition], dim=1)

            # Predict the noise using the model
            optimizer.zero_grad()
            predicted_noise = model(model_input, t)[:, :3]  # Take only the first 3 channels

            # Compute loss and backpropagate
            loss = loss_fn(predicted_noise, true_noise)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

        # Perform validation and save outputs in the desired folder
        validation_output_dir = os.path.join(output_dir, f"validation_plots_diffusion_{model_name}_{loss_fn_name}")
        best_val_loss = validate_diffusion_unet(model, diffusion, val_loader, device, validation_output_dir, epoch + 1, loss_fn, best_model_path, best_val_loss)

# Validation function
def validate_diffusion_unet(model, diffusion, dataloader, device, output_dir, epoch_number, loss_fn, best_model_path, best_val_loss):
    model.eval()
    total_loss = 0
    num_batches = 0

    # Clean the output directory
    if os.path.exists(output_dir):
        # Remove all existing files and subdirectories
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
    else:
        os.makedirs(output_dir)  # Create directory if it doesn't exist

    with torch.no_grad():
        for idx, (condition, target) in enumerate(dataloader):
            # Move tensors to device
            condition = condition.to(device)
            target = target.to(device)

            # Sample a random timestep
            t = torch.full((target.shape[0],), diffusion.num_timesteps - 1, dtype=torch.long, device=target.device)
            noisy_target = diffusion.q_sample(target, t)
            true_noise = (noisy_target - extract(diffusion.sqrt_alphas_cumprod, t, target.shape) * target) / extract(diffusion.sqrt_one_minus_alphas_cumprod, t, target.shape)
            
            # concatenate noisy target with condition
            model_input = torch.cat([noisy_target, condition], dim=1)

            # Predict the noise
            predicted_noise = model(model_input, t)[:, :3]  # Take only the first 3 channels

            # Compute loss
            loss = loss_fn(predicted_noise, true_noise)
            total_loss += loss.item()
            num_batches += 1

            # Reconstruct the denoised image
            reconstructed_target = (noisy_target - extract(diffusion.sqrt_one_minus_alphas_cumprod, t, target.shape) * predicted_noise) / extract(diffusion.sqrt_alphas_cumprod, t, target.shape)

            # Save predicted, ground truth, and conditioning images
            for i in range(condition.size(0)):
                denoised_image = (reconstructed_target[i].cpu().clamp(-1, 1) + 1) / 2  # Denormalize
                ground_truth = (target[i].cpu().clamp(-1, 1) + 1) / 2  # Denormalize
                condition_image = (condition[i].cpu().clamp(-1, 1) + 1) / 2  # Denormalize

                # Convert to numpy for saving
                denoised_np = (denoised_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                ground_truth_np = (ground_truth.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                condition_np = (condition_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Save the images
                epoch_folder = os.path.join(output_dir, f"epoch_{epoch_number}")
                os.makedirs(epoch_folder, exist_ok=True)

                Image.fromarray(denoised_np).save(os.path.join(epoch_folder, f"{idx}_{i}_denoised.png"))
                Image.fromarray(ground_truth_np).save(os.path.join(epoch_folder, f"{idx}_{i}_ground_truth.png"))
                Image.fromarray(condition_np).save(os.path.join(epoch_folder, f"{idx}_{i}_condition.png"))
    
    avg_val_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    print(f"Validation Loss (Epoch {epoch_number}): {avg_val_loss:.4f}")

    # Now `best_val_loss` is correctly tracked across epochs
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at Epoch {epoch_number} with validation loss: {best_val_loss:.4f}")

    return best_val_loss  # Return updated best validation loss


# Train function with real-time best model saving
def train_standard(model, train_loader, val_loader, optimizer, epochs, device, output_dir, loss_fn):
    model = model.to(device)

    # Get model and loss function names
    model_name = type(model).__name__
    loss_fn_name = loss_fn.__name__ if hasattr(loss_fn, '__name__') else loss_fn.__class__.__name__

    # Create directory for saving the model
    model_save_dir = os.path.join(output_dir, f"{model_name}_{loss_fn_name}")
    os.makedirs(model_save_dir, exist_ok=True)

    # Path to best model checkpoint
    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    best_val_loss = float("inf")  # Persist best loss across epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0  

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            recon_x = model(x)
            loss = loss_fn(recon_x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

        # Perform validation and save best model if needed
        validation_output_dir = os.path.join(output_dir, f"validation_plots_{model_name}_{loss_fn_name}")
        best_val_loss = validate_standard(model, val_loader, device, validation_output_dir, epoch + 1, loss_fn, best_model_path, best_val_loss)


# Standard validation with loss computation and best model saving
def validate_standard(model, dataloader, device, output_dir, epoch_number, loss_fn, best_model_path, best_val_loss):
    model.eval()
    total_loss = 0
    num_batches = 0

    # Clean the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            recon_x = model(x)
            loss = loss_fn(recon_x, y)
            total_loss += loss.item()
            num_batches += 1

            def denormalize(img):
                return ((img.cpu().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()

            epoch_folder = os.path.join(output_dir, f"epoch_{epoch_number}")
            os.makedirs(epoch_folder, exist_ok=True)

            for i in range(x.size(0)):
                input_img = denormalize(x[i])
                target_img = denormalize(y[i])
                output_img = denormalize(recon_x[i])

                Image.fromarray(input_img).save(os.path.join(epoch_folder, f"{idx}_{i}_input.png"))
                Image.fromarray(target_img).save(os.path.join(epoch_folder, f"{idx}_{i}_ground_truth.png"))
                Image.fromarray(output_img).save(os.path.join(epoch_folder, f"{idx}_{i}_output.png"))

    avg_val_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    print(f"Validation Loss (Epoch {epoch_number}): {avg_val_loss:.4f}")

    # Now `best_val_loss` is correctly tracked across epochs
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at Epoch {epoch_number} with validation loss: {best_val_loss:.4f}")

    return best_val_loss  # Return updated best validation loss


def main():
    args = parse_args()
    model_name = args.model.lower()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parameters
    timesteps = TIME_STEPS
    batch_size = args.batch      
    image_size = 256
    epochs = args.epochs
    device = DEVICE

    # Model selection 
    if args.model.lower() == 'unet':
        model = smp.Unet(
                        encoder_name="resnet34",       # encoder backbone
                        encoder_weights="imagenet",      # use pretrained encoder weights
                        in_channels=3,                 
                        classes=3                      # output channels (e.g., 3 for RGB)
                        )
    elif args.model.lower() == 'unet_diff':
        model = Unet(
                    dim=64,
                    channels=6,  # Adjusted for conditional input
                    dim_mults=(1, 2, 4, 8)
                    )
        diffusion = GaussianDiffusion(
                    model=model,
                    image_size=image_size,      # Adjust to your image resolution
                    timesteps=TIME_STEPS,     # Number of diffusion steps
                    )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss function selection 
    if args.loss.lower() == 'mse':
        loss_fn = F.mse_loss
    elif args.loss.lower() == 'weighted_mse':
        loss_fn = weighted_l2_loss
    elif args.loss.lower() == 'weighted_l1':
        loss_fn = weighted_l1_loss
    elif args.loss.lower() == 'ssim':
        loss_fn = ssim_loss
    elif args.loss.lower() == 'dice':
        loss_fn = dice_loss
    elif args.loss.lower() == 'weighted_dice':
        loss_fn = weighted_dice_loss
    elif args.loss.lower() == 'weighted_ssim':
        loss_fn = weighted_ssim_loss
    elif args.loss.lower() == 'perceptual':
        loss_fn = PerceptualLoss(device = device)
    elif args.loss.lower() == 'edge':
        loss_fn = edge_loss 
    elif args.loss.lower() == 'color':
        loss_fn = color_loss
    elif args.loss.lower() == 'wavelet':
        loss_fn = WaveletLoss(base_loss=ssim_loss, device = device)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    print(f"Using model: {args.model.lower()}")
    print(f"Using loss function: {args.loss.lower()}")
    print(f"Batch size: {args.batch}")
    print(f"Number of epochs: {args.epochs}")

    # Assuming raw_dataset is a list of (cond, x_0) PIL image pairs
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    # Initialize dataset
    dataset = TrajectoryDataset(image_dir=IMAGE_DIR, transform=transform)
    
    # Calculate the size of the subset to use and the training/validation split
    subset_size = int(DATASET_FRACTION * len(dataset))  # 40% of the dataset
    train_size = int((1 - VALIDATION_SPLIT) * subset_size)  # 80% of the subset for training
    val_size = subset_size - train_size  # 20% of the subset for validation

    # Randomly sample a subset of the dataset
    subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

    # Split the subset into training and validation sets
    train_dataset, val_dataset = random_split(subset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Train the model
    if args.model.lower() == 'unet':
        train_standard(model, train_loader, val_loader, optimizer, epochs, device, OUTPUT_DIR, loss_fn)
    elif args.model.lower() == 'unet_diff':
        train_diffusion_unet(model, diffusion, train_loader, val_loader, optimizer, epochs, device, OUTPUT_DIR, loss_fn)

if __name__ == '__main__':
    main()