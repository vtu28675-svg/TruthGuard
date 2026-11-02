
import pandas as pd
import os

# Dataset file paths
fake_path = "dataset/raw/Fake.csv"
true_path = "dataset/raw/True.csv"

# Check whether dataset files exist
if not os.path.exists(fake_path):
    print("Error: Fake.csv not found!")
    exit()

if not os.path.exists(true_path):
    print("Error: True.csv not found!")
    exit()

# Load datasets
fake_data = pd.read_csv(fake_path)
true_data = pd.read_csv(true_path)

# Add labels: 0 = Fake, 1 = Real
fake_data["label"] = 0
true_data["label"] = 1

# Combine both datasets
data = pd.concat([fake_data, true_data], ignore_index=True)

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display dataset information
print("TruthGuard Dataset Loaded Successfully!")
print("----------------------------------------")
print("Total News Articles:", len(data))
print("Fake News:", (data["label"] == 0).sum())
print("Real News:", (data["label"] == 1).sum())

print("\nDataset Preview:")
print(data[["title", "label"]].head())

# Save processed dataset
os.makedirs("dataset/processed", exist_ok=True)
data.to_csv("dataset/processed/news_dataset.csv", index=False)

print("\nProcessed dataset saved successfully!")
