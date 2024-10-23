import torch
import clip
from PIL import Image

# Reload the model structure and load weights from the saved file
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the model state from the saved file
model.load_state_dict(torch.load("clip_model.pth", map_location=device, weights_only=True))

# Set the model to evaluation mode
model.eval()

# Example image and text data
images = ["/Users/priya/PycharmProjects/multimodal_embeddings/table.png"]  # List of image paths
texts = ["What is the Flow rate at sect 12 to 307?"]  # List of corresponding texts

# Create lists to store embeddings
image_embeddings = []
text_embeddings = []

# Generate embeddings for the images
for img_path in images:
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_embeddings.append(image_features)

# Generate embeddings for the text
for text in texts:
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize
        text_embeddings.append(text_features)

# Assuming we want to query the first text and find its similarity with the first image
similarity = (100.0 * text_embeddings[0] @ image_embeddings[0].T).softmax(dim=-1)
print(f"Similarity score between the text and the image: {similarity}")

# Retrieve the top-k most similar images
_, indices = similarity[0].topk(k=1)  # Adjust k based on the number of embeddings
print(f"Top similar image index: {indices}")
