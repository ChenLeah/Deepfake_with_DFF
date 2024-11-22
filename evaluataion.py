import json
import os
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration
from peft import PeftModel
from sklearn.metrics import accuracy_score

# Define constants based on your preprocessing
LLAVA_IMAGE_TOKEN = "<|image|>"
IGNORE_INDEX = -1000000
EOT_TOKEN = "<|endoftext|>"
START_HEADER_TOKEN = "<|startofheader|>"
END_HEADER_TOKEN = "<|endofheader|>"

# Paths
test_json_path = "./testing_data.json"
image_dir = "./images"
output_results_path = "/home/users/nus/e1325997/scratch/output/llama3_lora_test_2/evaluation_results_final.json"
fine_tuned_model_path = "/home/users/nus/e1325997/scratch/output/llama3_lora_test_2/checkpoint-24000"
base_model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load the processor from the base model
processor = AutoProcessor.from_pretrained(base_model_name)

# Load the base model and apply the fine-tuned LoRA weights
base_model = MllamaForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)

# Functions for transforming data to match fine-tuning format
def replace_image_tokens(input_string):
    return input_string.replace("<image>", LLAVA_IMAGE_TOKEN)

def llava_to_openai(conversations):
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    for conversation in conversations:
        content = replace_image_tokens(conversation["value"])
        transformed_data.append({
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": [{"type": "text", "text": content}]
        })
    return transformed_data

# Function to extract the assistant's response
def extract_assistant_response(response_text):
    # Split response by assistant header to get the assistant's actual reply
    assistant_marker = "assistant\n"
    if assistant_marker in response_text:
        response_text = response_text.split(assistant_marker, 1)[-1]
    return response_text.strip()

# Evaluation
true_labels = []
predicted_labels = []
results = []

# Load test data
with open(test_json_path, "r") as f:
    test_data = json.load(f)

for item in test_data:
    image_filename = item["image"]
    image_path = os.path.join(image_dir, image_filename)
    
    # Transform conversations as per the fine-tuning data format
    prompt_data = llava_to_openai(item["conversations"][:-1])  # Remove the assistant's response from the prompt
    prompt_text = processor.apply_chat_template(prompt_data, add_generation_prompt=True)
    
    expected_answer = item["conversations"][1]["value"]
    true_label = "1" if "real" in expected_answer.lower() else "0"
    
    # Process image and prompt
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt_text, add_special_tokens=True, return_tensors="pt").to(model.device)
    
    # Generate model response
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)

    # Decode the output
    full_response = processor.decode(output[0], skip_special_tokens=True)
    response = extract_assistant_response(full_response)
    #print(response)
    predicted_label = "1" if "real" in response.lower() else "0"

    # Append results for each sample
    true_labels.append(int(true_label))
    predicted_labels.append(int(predicted_label))
    results.append({
        "id": item["id"],
        "prompt": prompt_text,
        "expected_answer": expected_answer,
        "model_output": response,
        "predicted_label": "Real" if predicted_label == "1" else "Deepfake"
    })

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the results to a JSON file
with open(output_results_path, "w") as outfile:
    json.dump(results, outfile, indent=4)
