"""
This script demonstrates creative text generation using Meta's Llama 2 70B model through AWS Bedrock.
It specifically generates Shakespeare-style poetry about machine learning, showcasing how to interact
with large language models via AWS's Bedrock service.

Key Features:
- Uses AWS Bedrock service to access Meta's Llama 2 70B model
- Implements proper prompt formatting with Llama 2-specific instruction tokens
- Provides configurable generation parameters:
    * max_gen_len: Controls the length of generated text
    * temperature: Adjusts randomness/creativity of outputs
    * top_p: Fine-tunes token selection diversity
- Demonstrates creative text generation in Shakespeare's literary style

Technical Requirements:
- AWS credentials must be properly configured in your environment
- boto3 library installed (AWS SDK for Python)
- Active subscription/access to AWS Bedrock service
- Appropriate IAM permissions for Bedrock service access

Usage:
The script can be run directly and will output a Shakespeare-style poem
about machine learning using the configured parameters.

Note: The model ID 'meta.llama3-70b-instruct-v1:0' represents the 70B parameter
version of Llama 2, which offers state-of-the-art text generation capabilities.
"""

import boto3  # AWS SDK for Python - Required for AWS service interaction
import json   # For JSON parsing and serialization of requests/responses

# Define the creative prompt for Shakespeare-style poetry generation
prompt_data = """
Act as a Shakespeare and write a poem on machine learning
"""

# Initialize the AWS Bedrock client with the runtime service
# This client will handle all communication with the AWS Bedrock API
bedrock = boto3.client(service_name="bedrock-runtime")

# Configure the model parameters for text generation with carefully tuned values
payload = {
    # Format prompt with Llama 2-specific instruction tokens
    # [INST] and [/INST] are special tokens that help the model:
    # - Clearly identify the instruction boundaries
    # - Maintain consistent response formatting
    # - Improve instruction-following behavior
    "prompt": "[INST]" + prompt_data + "[/INST]",

    # Maximum number of tokens to generate in the response
    # 512 tokens is approximately 350-400 words, suitable for a medium-length poem
    "max_gen_len": 512,

    # Temperature controls the randomness/creativity of the output:
    # - 0.0: Completely deterministic, always same output
    # - 0.5: Balanced creativity and coherence (recommended)
    # - 1.0: Maximum randomness, more creative but potentially less coherent
    "temperature": 0.5,

    # Top-p (nucleus) sampling parameter:
    # - Controls the cumulative probability threshold for token selection
    # - 0.9 means model considers tokens until their cumulative probability reaches 90%
    # - Higher values (>0.9) include more rare/creative words
    # - Lower values (<0.9) make output more focused and conservative
    "top_p": 0.9
}

# Convert the payload dictionary to a JSON string for API request
body = json.dumps(payload)

# Specify the Llama 2 model ID to use
# meta.llama3-70b-instruct-v1:0 is the 70B parameter version of Llama 2
model_id = "meta.llama3-70b-instruct-v1:0"

# Make the API call to AWS Bedrock for text generation
# The invoke_model method handles:
# 1. Authentication using configured AWS credentials
# 2. Request formatting and transmission
# 3. Response handling and error management
response = bedrock.invoke_model(
    # Convert our parameter dictionary to a JSON string
    body=body,
    # Specify the exact model version to use
    # This ensures consistent behavior across API updates
    modelId=model_id,
    # Define the response format we expect (JSON)
    accept="application/json",
    # Specify the format of our request body (JSON)
    contentType="application/json"
)

# Process the API response:
# 1. Get the response body (returns a StreamingBody object)
# 2. Read the stream into a string
# 3. Parse the JSON string into a Python dictionary
response_body = json.loads(response.get("body").read())

# Extract the generated text from the response
# The 'generation' field contains the model's output
# For Llama 2, this will be the Shakespeare-style poem
response_text = response_body['generation']

# Print the final generated Shakespeare-style poem about machine learning
print(response_text)
