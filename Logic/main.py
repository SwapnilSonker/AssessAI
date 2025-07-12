from groq import Groq
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.store_image import upload_to_cloudflare

load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
account_id = os.getenv("ACCOUNT_ID")

print(f"ACCOUNT_ID : {account_id}")

cloudflare_api_key = os.getenv("CLOUDFLARE_API_TOKEN")
print(f"cloudflare_api_key : {cloudflare_api_key}")

IMAGE_DATA_URL = "/Users/swapnilsonker/AssesaAI/Logic/image1.jpeg"

image_url = upload_to_cloudflare(IMAGE_DATA_URL ,cloudflare_api_key , account_id )

print(image_url)

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "system",
            "content": "You have to extract all the relevant contents from the image."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all the information from the image and show me in structured JSON format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": IMAGE_DATA_URL
                    }
                }
            ]
        }
    ],
    temperature=0.5,
    max_completion_tokens=1811,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},  # optional: defaults to plain text
    stop=None,
)

print(completion.choices[0].message.content)