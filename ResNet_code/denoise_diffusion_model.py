import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

# Diffusion model for denoising
class DiffusionModel:
    def __init__(self, timesteps=50, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        """Add noise to image x at timestep t."""
        alpha_bar = self.alpha_bars[t]
        noise = torch.randn_like(x)
        noisy_image = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
        return torch.clamp(noisy_image, 0, 1)  # Ensure values are in [0,1]

    def remove_noise(self, x, t):
        """Denoise image x by reversing the diffusion process."""
        alpha_bar = self.alpha_bars[t]
        denoised_image = x / torch.sqrt(alpha_bar)  # Reverse scaling
        return torch.clamp(denoised_image, 0, 1)

# Custom dataset class
class DFFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Save denoised images
def save_denoised_images(input_dir, output_dir, timestep=25, diffusion_timesteps=50):
    os.makedirs(output_dir, exist_ok=True)
    diffusion_model = DiffusionModel(timesteps=diffusion_timesteps)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = DFFDataset(root_dir=input_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for images, img_paths in tqdm(data_loader, desc="Processing images"):
        # Add noise to the image at the specified timestep
        noisy_image = diffusion_model.add_noise(images[0], timestep)
        
        # Denoise the image to reconstruct it
        denoised_image = diffusion_model.remove_noise(noisy_image, timestep)
        
        # Convert back to PIL and save the reconstructed image
        denoised_image = transforms.ToPILImage()(denoised_image)
        
        img_name = os.path.basename(img_paths[0])
        denoised_image.save(os.path.join(output_dir, img_name))

if __name__ == "__main__":
    input_dir = "images"  # Original images directory
    output_dir = "/home/users/nus/e1325997/scratch/denoised_images"  # Directory to save reconstructed (denoised) images
    save_denoised_images(input_dir, output_dir, timestep=25, diffusion_timesteps=50)
