import os
from transformers import CLIPModel, CLIPProcessor

# Step 1: Load the CLIP model and processor from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Step 2: Save the model and processor locally
model_save_path = "/Users/AMET-EEE/PycharmProjects/multimodal_ai/model" # Replace with Actual path
processor_save_path = "/Users/AMET-EEE/PycharmProjects/multimodal_ai/processor" # Replace with Actual path

# Create directories if they don't exist
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(processor_save_path, exist_ok=True)

# Save the model and processor
model.save_pretrained(model_save_path)
processor.save_pretrained(processor_save_path)

print("Model and processor saved locally.")