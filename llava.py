from groq import Groq
import base64
from dotenv import load_dotenv
import os
import mimetypes

load_dotenv()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "image.png"

# Getting the base64 string
base64_image = encode_image(image_path)

# Guess the MIME type of the image
mime_type, _ = mimetypes.guess_type(image_path)
if mime_type is None:
    # Default to octet-stream if type cannot be guessed
    mime_type = "application/octet-stream"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the text in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(chat_completion.choices[0].message.content)