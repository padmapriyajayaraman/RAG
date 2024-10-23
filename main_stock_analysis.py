import os
import base64
from PIL import Image
from io import BytesIO
import torch
import clip
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load the CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = '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'  # Replace with actual key in a secure way

# Initialize the LLM (GPT-4o)
llm = ChatOpenAI(model="gpt-4o")

# Function to convert image to Base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_image

# Query GPT-4o with the provided image and prompt
def query_gpt_with_image(image_path, query):
    base64_image = image_to_base64(image_path)

    # Formulate the prompt
    prompt = f"""
    You are a seasoned stock analyst specializing in swing trading for 10 to 15 days. Carefully analyze the image, 
    which contains a comprehensive technical analysis, including Fibonacci levels, Gann angles, SMA, EMA, and MACD. 
    The image title specifies the timeframe for plotting each line. Assess the chart starting from the most recent 
    price, as we are considering a new entry from this point onward. Identify potential entry and exit points that 
    ensure a minimum of 5% profit, providing specific values and a brief rationale. If no viable entry or exit is 
    found, explain why. Keep your response under 60 words.
    The user asked: "{query}".
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    )

    response = llm.invoke([message])
    return response

# Example usage
image_path = "/Users/priya/PycharmProjects/multimodal_embeddings/MANAPPURAM.png"
query = "What can you infer from this stock analysis?"
response = query_gpt_with_image(image_path, query)
print(response)
