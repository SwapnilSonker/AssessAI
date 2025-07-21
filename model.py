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
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"model : {model}")

for i in range(1):
    print(f"Input layer Norm: {model.model.layers[i].input_layernorm}")
    print(f"Post Attention Norm: {model.model.layers[i].post_attention_layernorm}")


# Access layer 0's attention query projection weights 
q_proj_weights_0 = model.model.layers[0].self_attn.q_proj.weight
print(f"q_proj_weights : {q_proj_weights_0}") 

k_proj_weights_0 = model.model.layers[0].self_attn.k_proj.weight
print(f"q_proj_weights_1 : {k_proj_weights_0}")

v_proj_weights_0 = model.model.layers[0].self_attn.v_proj.weight
print(f"q_proj_weights_2 : {v_proj_weights_0}")

o_proj_weights_0 = model.model.layers[0].self_attn.o_proj.weight
print(f"o_proj_weights_0 : {o_proj_weights_0}")

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

# output_1 = model.generate(**model_inputs , max_new_tokens = 1024)

# for i in range(0, 3):
# model.model.layers[0].mlp.act_fn = nn.ReLU()   # -> now modifying the layer 0 

# output_2 = model.generate(**model_inputs , max_new_tokens = 1024)

# print(f"output_1 : {tokenizer.decode(output_1[0], skip_special_tokens = True)}")
# print(f"output_2 : {tokenizer.decode(output_2[0], skip_special_tokens = True)}")

# print(f"model : {model}")
print(f"model layer 0 : {model.model.layers[0]}")
print(f"model lm_head : {model.lm_head}")

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(f"response : {response}")

# Get correct dtype from model
expected_dtype = model.model.layers[0].self_attn.v_proj.weight.dtype

# Create tensor with same dtype
hidden = torch.rand(1, 1, 2048, dtype=expected_dtype).to(model.device)

# Forward pass through attention projections
q = model.model.layers[0].self_attn.q_proj(hidden)
k = model.model.layers[0].self_attn.k_proj(hidden)
v = model.model.layers[0].self_attn.v_proj(hidden)

print(f"Q shape : {q.shape}")
print(f"K shape : {k.shape}")
print(f"V shape : {v.shape}")

with torch.no_grad():
    input_ids = model_inputs['input_ids']
    attention = model.model.layers[0].self_attn
    hidden_states = model.model.embed_tokens(input_ids)
    hidden_states = model.model.layers[0].input_layernorm(hidden_states)

    q = attention.q_proj(hidden_states)
    k = attention.k_proj(hidden_states)

    q_ = q[:, :, :256]  # Take only first 256 dims from Q to match K

    attn_scores = (q_ @ k.transpose(-2, -1)) / (q_.size(-1) ** 0.5)
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
    print("Attention probabilities (tokens attending to tokens):")
    print(attn_probs[0])
