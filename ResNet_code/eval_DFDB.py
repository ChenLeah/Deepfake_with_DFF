import argparse
import os
import torch
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score
from tqdm import tqdm
from PIL import Image

def load_model(model, model_path):
    state_dict = torch.load(model_path)
    
    # Handle the case where model was saved with DataParallel (i.e., with `module.` prefix)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model

# Dataset class for evaluation
class TestDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, limit=10):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Load real images
        real_images = sorted(os.listdir(real_dir))[:limit]
        self.image_paths.extend([os.path.join(real_dir, img) for img in real_images])
        self.labels.extend([0] * len(real_images))  # Label 0 for real

        # Load fake images
        fake_images = sorted(os.listdir(fake_dir))[:limit]
        self.image_paths.extend([os.path.join(fake_dir, img) for img in fake_images])
        self.labels.extend([1] * len(fake_images))  # Label 1 for fake

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image, label


# Evaluation metrics
def calculate_metrics(pred_probs, labels):
    preds = (pred_probs >= 0.5).astype(int)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, pred_probs) if len(np.unique(labels)) > 1 else float('nan')

    # Calculate EER
    fpr, tpr, thresholds = roc_curve(labels, pred_probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "eer": eer
    }


# Model evaluation
def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability of class 1 (fake)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(np.array(all_probs), np.array(all_labels))
    return metrics


# Model initialization based on argument
def initialize_model(model_name, model_path):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=False)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=False)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=False)
    else:
        raise ValueError(f"Invalid model name: {model_name}. Choose from 'resnet50', 'resnet101', or 'resnet152'.")

    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)

    return model


# Argument parser for the script
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned ResNet Models")
    parser.add_argument("--model_name", type=str, required=True, choices=["resnet50", "resnet101", "resnet152"], help="Specify the ResNet model variant")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory containing real images")
    parser.add_argument("--fake_dir", type=str, required=True, help="Directory containing fake images")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--limit", type=int, default=1000, help="Limit on the number of images per class")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model = initialize_model(args.model_name, args.model_path)
    model = load_model(model,args.model_path).to(device)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare the dataset and dataloader
    test_dataset = TestDataset(args.real_dir, args.fake_dir, transform=transform, limit=args.limit)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Evaluate the model
    metrics = evaluate_model(model, test_loader, device)

    # Print the evaluation metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"EER: {metrics['eer']:.4f}")


if __name__ == "__main__":
    main()
