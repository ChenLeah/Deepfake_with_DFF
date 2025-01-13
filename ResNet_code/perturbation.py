import os
import torch
import pandas as pd
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

# Individual perturbation functions

def load_model(model, model_path):
    state_dict = torch.load(model_path)
    
    # Handle the case where model was saved with DataParallel (i.e., with `module.` prefix)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model

def get_saturation_transform(factor=0.5):
    return transforms.Lambda(lambda img: ImageEnhance.Color(img).enhance(factor))

def get_contrast_transform(factor=0.5):
    return transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(factor))

def get_gaussian_blur_transform(radius=1):
    return transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius)))

def get_pixelation_transform(downscale_factor=0.1):
    def pixelate(img):
        original_size = img.size
        new_size = (int(original_size[0] * downscale_factor), int(original_size[1] * downscale_factor))
        img = img.resize(new_size, Image.NEAREST)
        return img.resize(original_size, Image.NEAREST)
    return transforms.Lambda(pixelate)

# Function to get the full transform pipeline with a specific perturbation
def get_transform_with_perturbation(perturbation):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        perturbation,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Custom dataset class with perturbations
class PerturbedDenoisedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.labels = [self.get_label(img) for img in os.listdir(root_dir)]
        
    def get_label(self, filename):
        if filename.startswith("wiki"):
            return 0
        elif filename.startswith("insight"):
            return 1
        elif filename.startswith("inpainting"):
            return 1
        elif filename.startswith("text2img"):
            return 1
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

# Calculate metrics
def calculate_metrics(preds, labels, num_classes=4):
    accuracy = accuracy_score(labels, preds)
    labels_bin = label_binarize(labels, classes=range(num_classes))
    preds_bin = label_binarize(preds, classes=range(num_classes))
    auc = roc_auc_score(labels_bin, preds_bin, average="macro", multi_class="ovr")
    
    eers = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], preds_bin[:, i])
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        eers.append(eer)
    mean_eer = np.mean(eers)
    
    return accuracy, auc, mean_eer

# Evaluate model for a specific perturbation
def evaluate_with_perturbation(model, data_loader, device):
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

# Load appropriate ResNet model based on name
def get_resnet_model(model_path):
    if 'ResNet50' in model_path:
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

# Main function to evaluate models and perturbations
def evaluate_models_and_perturbations(model_paths, denoised_image_dir, output_csv="robustness_evaluation_results.csv"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define all perturbations
    perturbations = {
        "saturation": get_saturation_transform(factor=0.5),
        "contrast": get_contrast_transform(factor=0.5),
        "gaussian_blur": get_gaussian_blur_transform(radius=1),
        "pixelation": get_pixelation_transform(downscale_factor=0.2)
    }
    
    results = []

    # Loop over each model path
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating model: {model_name}")
        
        # Load model
        model = get_resnet_model(model_path)
        model = load_model(model, model_path).to(device)

        # Loop over each perturbation
        for name, perturbation in perturbations.items():
            print(f"Applying perturbation: {name}")
            
            # Apply the perturbation-specific transform
            transform = get_transform_with_perturbation(perturbation)
            dataset = PerturbedDenoisedImageDataset(root_dir=denoised_image_dir, transform=transform)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

            # Evaluate the model with this perturbation
            accuracy, auc, mean_eer = evaluate_with_perturbation(model, data_loader, device)
            print(f"Results - Model: {model_name}, Perturbation: {name}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, EER: {mean_eer:.4f}")
            
            # Store the results
            results.append({
                "Model": model_name,
                "Perturbation": name,
                "Accuracy": accuracy,
                "AUC": auc,
                "EER": mean_eer
            })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nEvaluation results saved to {output_csv}")

# Run the evaluation
if __name__ == "__main__":
    model_paths = [
        "/home/users/nus/e1325997/scratch/ResNet_final/ResNet50_e10_b1024_lr0001/resnet50_dff_finetuned.pth",
        "/home/users/nus/e1325997/scratch/ResNet_final/ResNet101_e10_b128_lr0001/resnet101_dff_finetuned.pth",
        "/home/users/nus/e1325997/scratch/ResNet_final/ResNet152_e10_b128_lr0001/resnet152_dff_finetuned.pth",
    ]
    
    denoised_image_dir = "images"
    evaluate_models_and_perturbations(model_paths, denoised_image_dir)
