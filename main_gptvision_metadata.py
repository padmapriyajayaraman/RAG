import os
import base64
import fitz  # PyMuPDF
import torch
import clip
from PIL import Image
from io import BytesIO
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import TransformChain
from langchain_core.output_parsers import JsonOutputParser

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = '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'

# Load CLIP model for generating image embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

###########################################################
# Phase 1 - Comprehensive Metadata Generation for PDF
###########################################################

# Class for handling image metadata
class ImageInformation(BaseModel):
    image_description: str = Field(description="A short description of the schematic or image.")
    table_present: bool = Field(description="Whether there is a table in the image.")
    key_sections: list[str] = Field(description="List of key components or sections in the image.")
    connectivity_description: str = Field(description="Description of the pipeline or component connectivity.")
    line_types: dict = Field(description="Types of lines (e.g., dashed, constant) identified in the image.")

parser = JsonOutputParser(pydantic_object=ImageInformation)

# Function to convert an entire PDF page to a Base64-encoded image
def pdf_page_to_base64(pdf_path, page_number):
    pdf_document = fitz.open(pdf_path)  # Open the PDF document
    page = pdf_document.load_page(page_number - 1)  # Load the specific page (0-based index)
    pix = page.get_pixmap()  # Render the page as a pixmap (image)

    # Convert pixmap to an image and then to Base64 string
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffer = BytesIO()
    img.save(buffer, format="PNG")  # Save image in PNG format
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")  # Convert image to Base64 string

    return base64_image, pdf_document.page_count  # Return the Base64 string and total page count

# Function to extract comprehensive image information from the PDF
def extract_comprehensive_image_information(pdf_path, page_number):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)
    images = page.get_images(full=True)

    if images:
        # If there are images, process them
        for img in images:
            xref = img[0]  # Reference number for the image
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert image to Base64
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image = Image.open(BytesIO(base64.b64decode(img_base64)))

            # Generate CLIP embedding for the image
            image_embedding = generate_clip_embedding(image)

            # Use a comprehensive vision prompt to extract all relevant details from the image
            vision_prompt = f"""
            Analyze this image, which could contain schematics, tables, or diagrams. Extract the following:
            - Describe the key components or sections visible in the image.
            - Identify any labels or text present.
            - Determine if any tables are visible and briefly describe their content.
            - Explain the connectivity between components (e.g., pipelines, wiring).
            - Describe the types of lines visible (e.g., dashed, solid).
            """

            # Query the GPT-4o model with this comprehensive prompt
            llm = ChatOpenAI(model="gpt-4o")
            msg = llm.invoke(
                [HumanMessage(
                    content=[
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}]
                )]
            )

            # Parse the result and save the metadata
            image_metadata = parser.parse(msg.content)
            return image_metadata, image_embedding  # Return both metadata and embedding
    return None, None  # Return None if no images found

# Generate CLIP embedding for images
def generate_clip_embedding(input_data):
    if isinstance(input_data, Image.Image):
        image_input = preprocess(input_data).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        return image_features.cpu().numpy()

# Save metadata to a JSON file
def save_metadata(metadata, output_file):
    with open(output_file, 'w') as f:
        json.dump(metadata, f)

# Main function to generate comprehensive metadata from the PDF
def generate_pdf_metadata(pdf_path, output_folder):
    metadata_list = []
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count

    for page_number in range(1, total_pages + 1):
        page = pdf_document.load_page(page_number - 1)

        # Extract text content
        text_content = page.get_text("text")

        # Convert the entire page to Base64 for reference
        base64_image, _ = pdf_page_to_base64(pdf_path, page_number)

        # Handle images and extract metadata
        image_metadata, image_embedding = extract_comprehensive_image_information(pdf_path, page_number)

        # Store metadata for each page
        metadata = {
            "page_number": page_number,
            "text_content": text_content,
            "base64_image": base64_image,  # Store Base64 image for reference
            "image_metadata": image_metadata,  # Store comprehensive image metadata
            "image_embedding": image_embedding.tolist() if image_embedding is not None else None  # Image embedding
        }
        metadata_list.append(metadata)

    # Save the metadata to a file
    output_file = os.path.join(output_folder, "pdf_metadata.json")
    save_metadata(metadata_list, output_file)
    return metadata_list



###########################################################
# Phase 2 - Query Metadata Saved in Phase 1
###########################################################

# Load saved metadata
def load_metadata(metadata_file):
    with open(metadata_file, 'r') as f:
        return json.load(f)


# Function to process a user query based on pre-generated metadata
def process_query(metadata_file, query):
    metadata_list = load_metadata(metadata_file)

    for metadata in metadata_list:
        # Perform text-based search
        if query in metadata['text_content']:
            return metadata  # Return relevant page for the query

        # Search within image metadata for query
        if metadata['image_metadata']:
            if query in metadata['image_metadata'].image_description or query in metadata[
                'image_metadata'].key_sections:
                return metadata  # Return relevant image metadata

    return None  # Return None if no match is found


###########################################################
# Example Usage
###########################################################

pdf_path = "/Users/priya/PycharmProjects/multimodal_embeddings/diagram.pdf"
output_folder = "/Users/priya/PycharmProjects/multimodal_embeddings/metadata_vision"

# Generate comprehensive metadata by processing the PDF
generate_pdf_metadata(pdf_path, output_folder)

# Example query
metadata_file = os.path.join(output_folder, "pdf_metadata.json")
query = "How many pumps are there in the diagram?"
result = process_query(metadata_file, query)

# Output the result
print(result)
