# RAG: Image Retrieval Using CLIP Model

This repository, RAG, utilizes the CLIP (Contrastive Language–Image Pre-training) model from Hugging Face to perform image retrieval tasks focused on food, vegetables, and fruits. Users can input text queries related to these categories and retrieve the most relevant images.

# Description
RAG is designed to help users find images that best match given text queries related to food, vegetables, and fruits. The project leverages the CLIP model to generate embeddings for both the text queries and the images, allowing for efficient similarity scoring to identify the closest matches.

# Installation
Ensure you have Python 3.6 or higher. Install the required packages using pip:
pip install torch torchvision transformers matplotlib Pillow


# Usage
## Load and Save the Model:
The first part of the code loads the CLIP model and processor from Hugging Face and saves them locally. Update the model_save_path and processor_save_path variables to your desired local directories.

## Image Retrieval Process:
The next part of the code loads the model and processor, prepares a list of image paths, and computes similarities based on a text query.

## Run the Code:
Execute the script. It will output the similarity scores and display the top images that match the query.


# Reference
This project utilizes the Hugging Face Transformers library for loading the CLIP model. You can find the model details and documentation https://huggingface.co/


