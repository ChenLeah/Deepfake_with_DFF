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
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet model")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "resnet101", "resnet152"], help="Specify the ResNet model variant (resnet50, resnet101, or resnet152)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--root_dir", type=str, default="images", help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="/home/users/nus/e1325997/scratch/ResNet/", help="Output directory for saved model and logs")
    parser.add_argument("--output_name", type=str, default="resnet50_dff_finetuned.pth", help="Name of the saved model file")
    parser.add_argument("--log_name", type=str, default="training_log.txt", help="Name of the training log file")
    parser.add_argument("--seed", type=int, default=1234, help="set the random seed")
    args = parser.parse_args()
    return args


# Data transformations
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Custom dataset class
class DFFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.labels = [self.get_label(img) for img in os.listdir(root_dir)]
        
    def get_label(self, filename):
        if filename.startswith("wiki"):
            return 0  # Label for 'wiki'
        elif filename.startswith("insight"):
            return 1  # Label for 'insights'
        elif filename.startswith("inpainting"):
            return 2  # Label for 'inpainting'
        elif filename.startswith("text2img"):
            return 3  # Label for 'text2img'
        else:
            raise ValueError("Unknown prefix in filename")
    
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
def prepare_data(root_dir, batch_size=32, val_split=0.15, test_split=0.15):
    dataset = DFFDataset(root_dir=root_dir, transform=get_transform())
    train_size = int((1 - val_split - test_split) * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Define model and optimizer
def initialize_model(model_name="resnet50", lr=0.001):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    else:
        raise ValueError(f"Unknown model name {model_name}. Choose from 'resnet50', 'resnet101', or 'resnet152'.")

    # Modify the final layer to match the number of classes (4)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  # Four output classes

    # Multi-GPU support if needed
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def calculate_metrics(preds, labels, num_classes=4):
    # Initialize per-class metrics
    class_accuracies = []
    class_aucs = []
    class_eers = []
    
    # Calculate accuracy per class
    for class_id in range(num_classes):
        # Get binary labels for the current class (one-vs-rest)
        class_labels = (labels == class_id).astype(int)
        class_preds = (preds == class_id).astype(int)
        
        # Accuracy
        class_accuracy = accuracy_score(class_labels, class_preds)
        class_accuracies.append(class_accuracy)
        
        # AUC
        if len(np.unique(class_labels)) > 1:  # Check for non-trivial labels
            class_auc = roc_auc_score(class_labels, class_preds)
        else:
            class_auc = float('nan')  # Undefined AUC for single-class cases
        class_aucs.append(class_auc)
        
        # EER
        fpr, tpr, thresholds = roc_curve(class_labels, class_preds)
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        class_eers.append(eer)

    # Aggregate metrics
    overall_accuracy = accuracy_score(labels, preds)
    macro_auc = np.nanmean(class_aucs)
    macro_eer = np.nanmean(class_eers)
    
    return {
        "overall_accuracy": overall_accuracy,
        "macro_auc": macro_auc,
        "macro_eer": macro_eer,
        "class_accuracies": class_accuracies,
        "class_aucs": class_aucs,
        "class_eers": class_eers
    }


# Train and validate model, saving logs to file
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cuda', log_path="training_log.txt"):
    train_losses, val_losses = [], []
    with open(log_path, 'w') as log_file:
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

            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            running_val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

                    preds = outputs.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Calculate validation metrics
            metrics = calculate_metrics(np.array(val_preds), np.array(val_labels), num_classes=4)
            val_accuracy = metrics["overall_accuracy"]
            val_auc = metrics["macro_auc"]
            val_eer = metrics["macro_eer"]
            class_accuracies = metrics["class_accuracies"]
            class_aucs = metrics["class_aucs"]
            class_eers = metrics["class_eers"]

            # Log the results
            log_msg = (f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                       f"Val AUC: {val_auc:.4f}, Val EER: {val_eer:.4f}\n")
            log_file.write(log_msg)
            log_file.write("Per-class metrics:\n")
            for i in range(4):
                log_file.write(f"Class {i}: Accuracy: {class_accuracies[i]:.4f}, AUC: {class_aucs[i]:.4f}, EER: {class_eers[i]:.4f}\n")
            print(log_msg)

    return train_losses, val_losses


# Test model
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

    # Calculate metrics
    metrics = calculate_metrics(np.array(test_preds), np.array(test_labels), num_classes=4)
    test_accuracy = metrics["overall_accuracy"]
    test_auc = metrics["macro_auc"]
    test_eer = metrics["macro_eer"]
    class_accuracies = metrics["class_accuracies"]
    class_aucs = metrics["class_aucs"]
    class_eers = metrics["class_eers"]

    # Print and log the results
    print(f"Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}, Test EER: {test_eer:.4f}")
    print("Per-class metrics:")
    for i in range(4):
        print(f"Class {i}: Accuracy: {class_accuracies[i]:.4f}, AUC: {class_aucs[i]:.4f}, EER: {class_eers[i]:.4f}")


# Plot loss curves and save plot
def plot_losses(train_losses, val_losses, output_dir, plot_name="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss over Epochs")
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

def main():
    # Parse arguments
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data(args.root_dir, args.batch_size)

    # Initialize model, criterion, optimizer
    model, criterion, optimizer = initialize_model(model_name=args.model_name, lr=args.lr)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Train and validate the model
    log_path = os.path.join(args.output_dir, args.log_name)
    train_losses, val_losses = train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, args.epochs, 'cuda', log_path
    )

    # Plot and save loss graph
    plot_losses(train_losses, val_losses, args.output_dir, plot_name="loss_plot.png")

    # Test the model
    test_model(model, test_loader)

    # Save the fine-tuned model
    model_path = os.path.join(args.output_dir, args.output_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model fine-tuning completed and saved as '{model_path}'.")

# Run the script
if __name__ == "__main__":
    main()
