import os
# from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import sys
import argparse
from transformers.generation.utils import LogitsProcessorList
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)
from chatglm2_6b.configuration_chatglm import ChatGLMConfig
from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer
from onnx_export.utils import build_inputs
from transformers.models.bloom import BloomOnnxConfig
# from custom_op import *

from transformers import AutoTokenizer, AutoModel
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from accelerate import load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

parser = argparse.ArgumentParser(description='export pytorch model to onnx')
parser.add_argument(
    '--data_type',
    default="fp16",
    help='use fp16/fp32 to export onnx model. if use fp16, you need GPU memory > 24G, defualt is fp32'
)

args = parser.parse_args()
if args.data_type == "fp16":
    device = 'cuda'
else:
    device = 'cpu'

print(device)

output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
onnx_output_dir = os.path.join(output_dir, "onnx_output")
if not os.path.exists(onnx_output_dir):
    os.mkdir(onnx_output_dir)
else:
    for file in os.listdir(onnx_output_dir):
        os.remove(os.path.join(onnx_output_dir, file))
onnx_model_path = os.path.join(onnx_output_dir, "chatglm2_6b.onnx")

query = "想要出国留学，应该怎么办？"
history = [
    (
        "你好",
        "你好👋!我是人工智能助手 ChatGLM2-6B,很高兴见到你,欢迎问我任何问题。",
    )
]

# model_dir = os.path.join(project_dir, "chatglm2_6b")
# tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
# config = ChatGLMConfig.from_pretrained(model_dir)
# # config.num_layers = 1
# model = ChatGLMForConditionalGeneration.from_pretrained(model_dir, config=config)

model_name = '/home/faith/chatglm2-6b'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
max_mem = {0: 10_79532032, 1: 22_79532032, 'cpu': 912_90877440}
# model = load_checkpoint_and_dispatch(
#     model, '/home/faith/chatglm2-6b', device_map="balanced", no_split_module_classes=["GLMBlock"], offload_folder="/home/faith/langchain-ChatGLM/test", max_memory=max_mem
# )

# device_map = {'transformer.embedding': 0, 'transformer.rotary_pos_emb': 0, 'transformer.encoder.layers.0': 0, 'transformer.encoder.layers.1': 0, 'transformer.encoder.layers.2': 0, 'transformer.encoder.layers.3': 0, 'transformer.encoder.layers.4': 0, 'transformer.encoder.layers.5': 0, 'transformer.encoder.layers.6': 0, 'transformer.encoder.layers.7': 0, 'transformer.encoder.layers.8': 0, 'transformer.encoder.layers.9': 0, 'transformer.encoder.layers.10': 0, 'transformer.encoder.layers.11': 0, 'transformer.encoder.layers.12': 0, 'transformer.encoder.layers.13': 0, 'transformer.encoder.layers.14': 0, 'transformer.encoder.layers.15': 0, 'transformer.encoder.layers.16': 0, 'transformer.encoder.layers.17': 0, 'transformer.encoder.layers.18': 0, 'transformer.encoder.layers.19': 0, 'transformer.encoder.layers.20': 0, 'transformer.encoder.layers.21': 0, 'transformer.encoder.layers.22.input_layernorm': 0, 'transformer.encoder.layers.22.self_attention': 0, 'transformer.encoder.layers.22.post_attention_layernorm': 0, 'transformer.encoder.layers.23': 1, 'transformer.encoder.layers.24.input_layernorm': 1, 'transformer.encoder.layers.24.post_attention_layernorm': 1, 'transformer.encoder.layers.24.mlp': 1, 'transformer.encoder.layers.25': 1, 'transformer.encoder.layers.26': 1, 'transformer.encoder.layers.27': 1, 'transformer.encoder.final_layernorm': 1, 'transformer.output_layer': 1, 'transformer.encoder.layers.24.self_attention': 1, 'transformer.encoder.layers.22.mlp': 1}

device_map = {'transformer.embedding': 0, 'transformer.rotary_pos_emb': 0, 'transformer.encoder.layers.0': 0, 'transformer.encoder.layers.1': 0, 'transformer.encoder.layers.2': 0, 'transformer.encoder.layers.3': 0, 'transformer.encoder.layers.4': 0, 'transformer.encoder.layers.5': 0, 'transformer.encoder.layers.6': 0, 'transformer.encoder.layers.7': 0, 'transformer.encoder.layers.8': 0, 'transformer.encoder.layers.9': 0, 'transformer.encoder.layers.10': 0, 'transformer.encoder.layers.11': 0, 'transformer.encoder.layers.12': 0, 'transformer.encoder.layers.13': 0, 'transformer.encoder.layers.14': 0, 'transformer.encoder.layers.15': 0, 'transformer.encoder.layers.16': 0, 'transformer.encoder.layers.17': 0, 'transformer.encoder.layers.18': 0, 'transformer.encoder.layers.19': 0, 'transformer.encoder.layers.20': 0, 'transformer.encoder.layers.21': 0, 'transformer.encoder.layers.22': 0, 'transformer.encoder.layers.23': 1, 'transformer.encoder.layers.24': 1, 'transformer.encoder.layers.25': 0, 'transformer.encoder.layers.26': 1, 'transformer.encoder.layers.27': 1, 'transformer.encoder.final_layernorm': 1, 'transformer.output_layer': 1}

offload_state_dict = False
load_checkpoint_in_model(
    model,
    checkpoint='/home/faith/chatglm2-6b',
    device_map=device_map,
    offload_folder="/home/faith/langchain-ChatGLM/test",
    dtype=None,
    offload_state_dict=offload_state_dict,
    offload_buffers=False,
)

model = dispatch_model(
    model,
    device_map=device_map,
    offload_dir="/home/faith/langchain-ChatGLM/test",
    offload_buffers=False,
    preload_module_classes=None,
)


if device == "cuda":
    # model = model.half().cuda()
    model = model.float()
else:
    model = model.float().cpu()
device = torch.device('cuda:0')
model.eval()
# input_tensors
input_tensors = build_inputs(device, tokenizer, query, history)

# --debug for chat --
# response, history = model.chat(tokenizer, query, history)
# print("res", response)

print(" ---forward first --- ")
outputs = model.forward(
    **input_tensors
)

print("--second forward ---")
# input_ids = input_tensors["input_ids"]
attention_mask = input_tensors["attention_mask"]
position_ids = input_tensors["position_ids"]
past_key_values = outputs["past_key_values"]
# copy from forward in second time
input_ids = torch.tensor([[30910]]).to(device)

# copy from _update_model_kwargs_for_generation in modeling_chatglm.py
attention_mask = torch.cat(
    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
)
new_position_id = position_ids[..., -1:].clone()
new_position_id += 1
position_ids = torch.cat(
    [position_ids, new_position_id], dim=-1
)
# copy from prepare_inputs_for_generation in modeling_chatglm.py
position_ids = position_ids[..., -1:]
# print shape
print(
    "input_ids shape:", input_ids.shape,
    "; type:", input_ids.dtype
)
print(
    "position_ids shape:", position_ids.shape,
    "; type: ", input_ids.dtype
)
print(
    "attention_mask shape:", attention_mask.shape,
    "; type: ", attention_mask.dtype
)
print(
    "one past_key_value shape: ", past_key_values[0][0].shape,
    "; type:", past_key_values[0][0].dtype
)
print("logits shape: ", outputs["logits"].shape)
outputs2 = model.forward(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values
)



print("--- export onnx ---")
# ---prepare for onnx export ---
input_names = ["input_ids", 'position_ids', "attention_mask"]
output_names = ["logits"]
dynamic_axes = {
    'input_ids': {0: "batch_size", 1: "sequence"},
    'position_ids': {0: "batch_size", 1: "sequence"},
    "attention_mask": {0: "batch_size", 1: "past_sequence + sequence"},
    "logits": {0: "batch_size", 1: "sequence"}
}
for layer_idx in range(model.config.num_layers):
    # --- input key and value ---
    past_key_name = f"past_key_values.{layer_idx}.key"
    past_value_name = f"past_key_values.{layer_idx}.value"
    input_names += [past_key_name, past_value_name]
    # --- output key and value ---
    present_key_name = f"present_key_values.{layer_idx}.key"
    present_value_name = f"present_key_values.{layer_idx}.value"
    output_names += [present_key_name, present_value_name]
    dynamic_axes.update({
        past_key_name: {
            0: "past_sequence",
            1: "batch_size",
        },
        past_value_name: {
            0: "past_sequence",
            1: "batch_size",
        },
        present_key_name: {
            0: "past_sequence + 1",
            1: "batch_size"
        },
        present_value_name: {
            0: "past_sequence + 1",
            1: "batch_size"
        }
    })


torch_gc()
# model = model.cpu().float()

# input_ids = input_ids.cpu()
# position_ids = position_ids.cpu()
# attention_mask = attention_mask.cpu()

# for t in past_key_values:
#     if torch.is_tensor(t):
#         t = t.cpu()
# past_key_values = past_key_values.cpu()

with torch.no_grad():
    torch.onnx.export(
        model,
        args=(
            input_ids,
            position_ids,
            attention_mask, 
            past_key_values
        ),
        f=onnx_model_path,
        # custom_opsets = {"torch.onnx": 17},
        # opset_version=18,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )


# /mnt/ChatGLM2-6B-TensorRT/chatglm2_6b/modeling_chatglm.py:823: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
# /mnt/ChatGLM2-6B-TensorRT/chatglm2_6b/modeling_chatglm.py:223: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:

# /mnt/ChatGLM2-6B-TensorRT/chatglm2_6b/modeling_chatglm.py:951: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   if return_last_logit:




# https://github.com/Tlntin/ChatGLM2-6B-TensorRT/issues/13  2.1.0

# # Linux, CPU only
# pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# # Linux, CUDA 11.7
# pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu117/torch_nightly.html

# # Linux, CUDA 11.8  use this
# pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html

# export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64/:/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/nvidia/cudnn/lib:/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/torch/lib::/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/tensorrt
# pip install transformers==4.30.2

# # Linux, ROCm 5.4.2
# pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/rocm5.4/torch_nightly.html

# # Mac, CPU only
# pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# # Windows, CPU only
# pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# # Windows, CUDA 11.7
# pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu117/torch_nightly.html

# # Windows, CUDA 11.8
# pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html
