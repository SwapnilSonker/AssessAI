# from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch.nn as nn
import torch
from huggingface_hub import snapshot_download

# To know the local location of the model downloaded
# local_model_path = snapshot_download("Qwen/Qwen2.5-3B-Instruct")
# print("📁 Model is stored at:", local_model_path)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print(f"model : {model}")


# Access layer 0's attention query projection weights 
# q_proj_weights_0 = model.model.layers[0].self_attn.q_proj.weight
# print(f"q_proj_weights : {q_proj_weights_0}") 

# q_proj_weights_1 = model.model.layers[1].self_attn.q_proj.weight
# print(f"q_proj_weights_1 : {q_proj_weights_1}")

# q_proj_weights_2 = model.model.layers[2].self_attn.q_proj.weight
# print(f"q_proj_weights_2 : {q_proj_weights_2}")

# <---- The below is used to print the self attention weights of the model ----->
# for i in range(0, 35):
#     q_proj_weights = model.model.layers[i].self_attn.q_proj.weight
#     print(f"q_proj_weights_{i} : {q_proj_weights}")


tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

output_1 = model.generate(**model_inputs , max_new_tokens = 1024)

for i in range(0, 3):
    model.model.layers[i].mlp.act_fn = nn.ReLU()   # -> now modifying the layer 0 

output_2 = model.generate(**model_inputs , max_new_tokens = 1024)

print(f"output_1 : {tokenizer.decode(output_1[0], skip_special_tokens = True)}")
print(f"output_2 : {tokenizer.decode(output_2[0], skip_special_tokens = True)}")

print(f"model : {model}")

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(f"response : {response}")
