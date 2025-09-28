"""
This script demonstrates the usage of AWS Bedrock's Stable Diffusion XL model for image generation.
It uses the AWS SDK (boto3) to interact with the Bedrock service and generates high-quality images 
based on text prompts using the Stable Diffusion XL model.

Features:
    - Text-to-image generation using AWS Bedrock
    - Customizable image parameters (size, quality, etc.)
    - Automatic output directory creation
    - Base64 image encoding/decoding
    
Requirements:
    - boto3: AWS SDK for Python
    - AWS credentials configured with Bedrock access
    - Access to AWS Bedrock service and Stable Diffusion XL model
    
Usage:
    1. Configure AWS credentials
    2. Modify the prompt_data variable with desired image description
    3. Adjust model parameters if needed (cfg_scale, steps, size)
    4. Run the script to generate and save the image
"""

# Import required libraries
import boto3      # AWS SDK for Python
import json       # For JSON serialization/deserialization
import base64    # For encoding/decoding image data
import os        # For file and directory operations

# Define the image generation prompt text
prompt_data = """
provide me an 4k hd image of a beach, also use a blue sky rainy season and
cinematic display
"""

# Format the prompt for the model
# weight: Controls the importance of this prompt (1.0 is standard weight)
prompt_template = [{"text": prompt_data, "weight": 1}]

# Initialize the AWS Bedrock runtime client
# This client will handle all communication with the Stable Diffusion XL model
bedrock = boto3.client(service_name="bedrock-runtime")

# Define the model generation parameters
payload = {
    "text_prompts": prompt_template,  # List of text prompts with their weights
    # Classifier Free Guidance scale (7-14 recommended, higher = more prompt adherence)
    "cfg_scale": 10,
    # Random seed for reproducibility (0 = random seed each time)
    "seed": 0,
    # Number of diffusion steps (higher = better quality but slower)
    "steps": 50,
    # Output image width in pixels (must be multiple of 8)
    "width": 512,
    # Output image height in pixels (must be multiple of 8)
    "height": 512
}

# Convert payload dictionary to JSON string for API request
body = json.dumps(payload)

# Specify the Stable Diffusion XL model ID in AWS Bedrock
# Latest version of Stable Diffusion XL
model_id = "stability.stable-diffusion-xl-v1"

# Call the AWS Bedrock API to generate the image
# This makes a synchronous request that will block until the image is generated
response = bedrock.invoke_model(
    body=body,                          # The JSON string containing generation parameters
    modelId=model_id,                   # The specific model to use
    accept="application/json",          # Expected response format
    contentType="application/json",     # Request payload format
)

# Parse the JSON response from the API
response_body = json.loads(response.get("body").read())
# Print the full response for debugging purposes
print(response_body)  # Contains metadata about the generation process

# Extract the generated image data from the response
# The API returns an array of artifacts, we take the first one
artifact = response_body.get("artifacts")[0]
# Get the base64-encoded image data and encode it to UTF-8
image_encoded = artifact.get("base64").encode("utf-8")

# Decode the base64 image data into binary bytes
image_bytes = base64.b64decode(image_encoded)

# Set up the output directory and file handling
output_dir = "output"
# Create the output directory if it doesn't exist, ignore if it does
os.makedirs(output_dir, exist_ok=True)
# Define the output filename with path
file_name = f"{output_dir}/generated-img.png"
# Save the binary image data as a PNG file
with open(file_name, "wb") as f:
    f.write(image_bytes)
