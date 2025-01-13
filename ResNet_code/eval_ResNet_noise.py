import os
import torch
import pandas as pd
import random
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import numpy as np
from sklearn.preprocessing import label_binarize


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function to load model with appropriate state_dict handling
def load_model(model, model_path):
    state_dict = torch.load(model_path)
    
    # Handle the case where model was saved with DataParallel (i.e., with `module.` prefix)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model

# Custom dataset class for loading denoised images
class DFFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.labels = [self.get_label(img) for img in os.listdir(root_dir)]
        
    def get_label(self, filename):
        if filename.startswith("wiki"):
            return 0  # Label for 'real'
        else:
            return 1  # Label for 'fake'
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Prepare data loaders
def prepare_data(root_dir, batch_size=512, val_split=0.15, test_split=0.15):
    dataset = DFFDataset(root_dir=root_dir, transform=get_transform())
    train_size = int((1 - val_split - test_split) * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Calculate metrics
def calculate_metrics(preds, labels, num_classes=2):
    accuracy = accuracy_score(labels, preds)
    labels_bin = label_binarize(labels, classes=range(num_classes))
    preds_bin = label_binarize(preds, classes=range(num_classes))
    
    auc = None
    mean_eer = None
    eers = []

    if labels_bin.shape[1] > 1:  # Ensure there is more than one class
        auc = roc_auc_score(labels_bin, preds_bin, average="macro", multi_class="ovr")
        
        # EER Calculation per class
        for i in range(labels_bin.shape[1]):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], preds_bin[:, i])
            eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
            eers.append(eer)
        mean_eer = np.mean(eers)
    else:
        auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else float('nan')
        fpr, tpr, thresholds = roc_curve(labels, preds)
        mean_eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))] if len(fpr) > 0 else float('nan')
        # Handle single-class case
        print("Only one class detected in the data. AUC and EER cannot be calculated.")
    
    return accuracy, auc, mean_eer

# Evaluate a single model and return metrics
def evaluate_single_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating model"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return calculate_metrics(np.array(all_preds), np.array(all_labels))

# Load appropriate ResNet model based on path
def get_resnet_model(model_path):
    if ('ResNet50' in model_path) or ('resnet50' in model_path):
        model = models.resnet50(pretrained=False)
    elif 'ResNet101' in model_path:
        model = models.resnet101(pretrained=False)
    elif 'ResNet152' in model_path:
        model = models.resnet152(pretrained=False)
    else:
        raise ValueError("Model path does not specify a valid ResNet variant (50, 101, or 152).")
    
    # Adjust the final layer for 4-class classification
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    return model

# Main function to evaluate models from a list of paths and store results
def evaluate_models_from_paths(model_paths, denoised_image_dir, output_csv="evaluation_results.csv"):
    # Define transformation for evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load denoised image dataset
    set_seed()
    train_loader, val_loader, test_loader = prepare_data(denoised_image_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    # Iterate over all model paths
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"Evaluating model: {model_name}")

        # Load the appropriate ResNet model variant
        model = get_resnet_model(model_path)
        model = load_model(model, model_path)
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(device)
        # Evaluate model
        accuracy, auc, mean_eer = evaluate_single_model(model, test_loader, device)
        print(f"Results for {model_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, EER: {mean_eer:.4f}")
        
        # Store the results
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "AUC": auc,
            "EER": mean_eer
        })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Evaluation results saved to {output_csv}")


if __name__ == "__main__":
    # List of model paths
    model_paths = [
        "/home/users/nus/e1325997/scratch/ResNet_binary/resnet50_pixelation_contrast_e1_b512_lr00001/final_model.pth",
    ]
    
    denoised_image_dir = "/home/users/nus/e1325997/Llama3.2-Vision-Finetune/images"
    evaluate_models_from_paths(model_paths, denoised_image_dir, "eval_results_adver2.csv")
