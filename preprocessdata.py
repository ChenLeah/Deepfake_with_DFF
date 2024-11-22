import os
import json
import shutil
from pathlib import Path

# Define the root data directory and output paths
data_dir = Path('./data')
output_dir = Path('./images')
output_dir.mkdir(exist_ok=True)

json_output_path = './dataset.json'
json_data = []

# Define folders and labels
folders = {
    "wiki": "Real",
    "inpainting": "Deepfake",
    "insight": "Deepfake",
    "text2img": "Deepfake"
}

# Process each folder, flatten and rename images, and create JSON entries
for folder_name, answer in folders.items():
    folder_path = data_dir / folder_name
    for subfolder in folder_path.iterdir():
        if subfolder.is_dir():
            for image_file in subfolder.iterdir():
                if image_file.is_file():
                    # Define new image name with prefix and copy to output folder
                    new_image_name = f"{folder_name}_{image_file.stem}.jpg"
                    new_image_path = output_dir / new_image_name
                    shutil.copy(image_file, new_image_path)

                    # Create JSON entry
                    json_entry = {
                        "id": f"{folder_name}_{image_file.stem}",
                        "image": new_image_name,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\nIs this a real photo or a deepfake photo?"
                            },
                            {
                                "from": "gpt",
                                "value": answer
                            }
                        ]
                    }
                    json_data.append(json_entry)

# Save JSON data to file
with open(json_output_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

output_dir, json_output_path
