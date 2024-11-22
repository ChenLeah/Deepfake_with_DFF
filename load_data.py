import requests
import zipfile
import os

# Define URLs and destination folders
dataset_urls = {
    "wiki": "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/wiki.zip",
    "inpainting": "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/inpainting.zip",
    "insight": "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/insight.zip",
    "text2img": "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/text2img.zip"
}
destination_path = "./data"  # Destination folder to save unzipped files

# Create destination path if not exists
os.makedirs(destination_path, exist_ok=True)

for folder_name, url in dataset_urls.items():
    print(f"Downloading {folder_name}...")
    response = requests.get(url, stream=True)
    zip_path = os.path.join(destination_path, f"{folder_name}.zip")
    
    # Save zip file
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination_path)
    
    # Remove the zip file after extraction
    os.remove(zip_path)
    print(f"{folder_name} downloaded and extracted.")

print("All folders downloaded and extracted successfully.")
