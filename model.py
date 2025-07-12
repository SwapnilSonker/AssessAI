from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch

# Load processor and model
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an expert in data extraction and you have to extract all the relevant contents from the image."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "path": "/Users/swapnilsonker/AssesaAI/Logic/image1.jpeg"
            },
            {
                "type": "text",
                "text": "Describe this image in detail"
            }
        ]
    }
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = True,
    return_dict = True,
    return_tensors = "pt"
).to(device)

generated_ids = model.generate(**inputs, max_new_tokens = 512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids , generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed , skip_special_tokens = True 
)[0]

print(f"output text : {output_text}")

