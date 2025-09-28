# Import required libraries
import boto3  # AWS SDK for Python
import json   # For JSON processing

# Define the prompt for the AI model
# This prompt asks the model to write a Shakespearean-style poem about Generative AI
prompt_data = """
Act as a Shakespeare and write a poem on Genertaive AI
"""

# Initialize the AWS Bedrock client for accessing AI models
bedrock = boto3.client(service_name="bedrock-runtime")

# Configure the request payload for the AI model
payload = {
    "max_tokens": 512,      # Maximum number of tokens in the response
    # Controls randomness (0=deterministic, 1=more random)
    "temperature": 0.7,
    "top_p": 0.9,          # Controls diversity of word choices
    'messages': [           # List of conversation messages
        {
            'role': 'user',           # Message from the user
            'content': prompt_data    # The actual prompt we defined earlier
        }
    ],
}
# Convert the payload to JSON string
body = json.dumps(payload)

# Specify which AI model to use
model_id = "ai21.jamba-instruct-v1:0"

# Make the API call to the AI model
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

# Process the response
response_body = json.loads(response.get("body").read())  # Parse JSON response
response_text = response_body.get("choices")[0].get(
    "message").get("content")  # Extract generated text

# Print the AI-generated response
print(response_text)
