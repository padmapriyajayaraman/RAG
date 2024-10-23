import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt

# Step 1: Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 2: Load the model and processor from the saved paths
model_save_path = "/Users/AMET-EEE/PycharmProjects/multimodal_ai/model"  # Replace with your actual path
processor_save_path = "/Users/AMET-EEE/PycharmProjects/multimodal_ai/processor"  # Replace with your actual path

model = CLIPModel.from_pretrained(model_save_path).to(device)
processor = CLIPProcessor.from_pretrained(processor_save_path)

# Step 3: Set the model to evaluation mode
model.eval()

# List of image paths
image_paths = [
    "/Users/AMET-EEE/PycharmProjects/multimodal_ai/image_1.jfif",
    "/Users/AMET-EEE/PycharmProjects/multimodal_ai/image_2.jfif",
    "/Users/AMET-EEE/PycharmProjects/multimodal_ai/image_3.jfif"
]

# Text query
text_query = "Find me a picture of green apple."

# Create lists to store embeddings
image_embeddings = []

# Generate embeddings for the images
for img_path in image_paths:
    image = Image.open(img_path)  # Open the image
    # Use the processor to prepare the image
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_embeddings.append(image_features)

# Generate embeddings for the text
text_inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

# Calculate similarities
similarities = []
for img_feature in image_embeddings:
    img_feature /= img_feature.norm(dim=-1, keepdim=True)  # Normalize
    raw_similarity = (text_features @ img_feature.T)  # Calculate raw similarity
    similarities.append(raw_similarity.item())  # Store the raw similarity score

# After calculating, print the raw similarity scores
print("Raw Similarity Scores:", similarities)

# Retrieve the top-k most similar images
top_k = 3  # Change this value to get more/less top results
top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

# Display results
print(f"Top {top_k} similar images for the query '{text_query}':")
for index in top_indices:
    print(f"Image: {image_paths[index]}, Similarity Score: {similarities[index]:.2f}")

# Visualize the top similar images
plt.figure(figsize=(12, 6))
for i, index in enumerate(top_indices):
    plt.subplot(1, top_k, i + 1)
    img = Image.open(image_paths[index])
    plt.imshow(img)
    plt.title(f"Score: {similarities[index]:.2f}")
    plt.axis("off")
plt.show()
