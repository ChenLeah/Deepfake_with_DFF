import argparse
import os
import torch
import numpy as np
import random
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from PIL import Image, ImageEnhance

# Define randomized perturbation transforms
def get_randomized_contrast_transform(min_factor=0.5, max_factor=1.5):
    return transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(min_factor, max_factor)))

def get_randomized_pixelation_transform(min_factor=0.1, max_factor=0.3):
    def pixelate(img):
        downscale_factor = random.uniform(min_factor, max_factor)
        original_size = img.size
        new_size = (int(original_size[0] * downscale_factor), int(original_size[1] * downscale_factor))
        img = img.resize(new_size, Image.NEAREST)
        return img.resize(original_size, Image.NEAREST)
    return transforms.Lambda(pixelate)

def get_randomized_gaussian_noise_transform(min_std=0.05, max_std=0.1):
    def add_gaussian_noise(img):
        tensor = transforms.ToTensor()(img)  # Convert PIL to Tensor temporarily
        tensor = tensor + torch.randn_like(tensor) * random.uniform(min_std, max_std)
        return transforms.ToPILImage()(tensor.clamp(0, 1))  # Convert back to PIL image
    return transforms.Lambda(add_gaussian_noise)


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load model weights
def load_model(model, model_path):
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Fine-tuning with Perturbations")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "resnet101", "resnet152"], help="Specify the ResNet model variant")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--root_dir", type=str, default="images", help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saved model and logs")
    parser.add_argument("--perturbation", type=str, default="contrast", choices=["contrast", "pixelation", "gaussian_noise"], help="Type of perturbation to apply")
    parser.add_argument("--fine_tuned_path", type=str, default=None, help="Path to the fine-tuned model weights (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

# Define perturbation transforms
def get_perturbation_transform(perturbation):
    if perturbation == "contrast":
        return get_randomized_contrast_transform(min_factor=0.5, max_factor=1.5)
    elif perturbation == "pixelation":
        return get_randomized_pixelation_transform(min_factor=0.1, max_factor=0.4)
    elif perturbation == "gaussian_noise":
        return get_randomized_gaussian_noise_transform(min_std=0.05, max_std=0.15)
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")

# Data transformations
def get_transform(perturbation):
    perturb_transform = get_perturbation_transform(perturbation)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        perturb_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Custom dataset class
class PerturbedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.labels = [self.get_label(img) for img in os.listdir(root_dir)]
    
    def get_label(self, filename):
        return 0 if filename.startswith("wiki") else 1  # 0: real, 1: fake
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Prepare data loaders
def prepare_data(root_dir, batch_size, perturbation, val_split=0.15, test_split=0.15):
    dataset = PerturbedDataset(root_dir, transform=get_transform(perturbation))
    train_size = int((1 - val_split - test_split) * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Define model and optimizer
def initialize_model(model_name, lr, fine_tuned_path=None):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification (real vs. fake)
    # model = nn.DataParallel(model)

    if fine_tuned_path and os.path.exists(fine_tuned_path):
        print(f"Loading fine-tuned model weights from: {fine_tuned_path}")
        model = load_model(model, fine_tuned_path)
    else:
        print("Training from scratch or pre-trained weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

# Metrics calculation
def calculate_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else float('nan')
    fpr, tpr, thresholds = roc_curve(labels, preds)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))] if len(fpr) > 0 else float('nan')
    return {"accuracy": accuracy, "auc": auc, "eer": eer}

# Training loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        metrics = calculate_metrics(np.array(val_preds), np.array(val_labels))
        print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, EER: {metrics['eer']:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))

def test_model(model, test_loader, device='cuda'):
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(np.array(test_preds), np.array(test_labels))
    print(f"Test Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, EER: {metrics['eer']:.4f}")

# Main function
def main():
    args = parse_args()
    set_seed(args.seed)
    train_loader, val_loader, test_loader = prepare_data(args.root_dir, args.batch_size, args.perturbation)
    model, criterion, optimizer = initialize_model(args.model_name, args.lr, args.fine_tuned_path)
    model = nn.DataParallel(model)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, args.epochs, 'cuda', args.output_dir)
    test_model(model, test_loader)
    model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model fine-tuning completed and saved as '{model_path}'.")

if __name__ == "__main__":
    main()
