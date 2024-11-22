import random
import json

# Load the prepared dataset
with open('dataset.json', 'r') as json_file:
    data = json.load(json_file)

# Randomize the dataset entries
random.shuffle(data)

# Define split ratio for training and testing (e.g., 80% training, 20% testing)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)

# Split data
training_data = data[:split_index]
testing_data = data[split_index:]

# Save the training and testing datasets
with open('training_data.json', 'w') as train_file:
    json.dump(training_data, train_file, indent=2)

with open('testing_data.json', 'w') as test_file:
    json.dump(testing_data, test_file, indent=2)
