from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

IMAGE_DATA_URL = "/Users/swapnilsonker/AssesaAI/Logic/Untitled document.pdf"

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