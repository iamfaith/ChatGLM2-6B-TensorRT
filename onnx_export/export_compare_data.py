import os
import sys
import torch
import argparse

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)
from onnx_export.utils import build_inputs
from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer
from chatglm2_6b.configuration_chatglm import ChatGLMConfig
parser = argparse.ArgumentParser(description='export pytorch model to onnx')
parser.add_argument(
    '--data_type',
    default="fp32",
    help='use fp16/fp32 to export input/output, Defualt is fp32'
)

args = parser.parse_args()
if args.data_type == "fp16":
    device = 'cuda'
else:
    device = 'cpu'

output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save input tensor
pt_input_path1 = os.path.join(output_dir, "pt_input1.pt")
pt_input_path2 = os.path.join(output_dir, "pt_input2.pt")
pt_input_dict1 = dict()
pt_input_dict2 = dict()
# save output tensor
pt_output_path1 = os.path.join(output_dir, "pt_output1.pt")
pt_output_path2 = os.path.join(output_dir, "pt_output2.pt")
pt_output_dict1 = dict()
pt_output_dict2 = dict()


class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])


query = "晚上睡不着应该怎么办"
history = [
    (
        "你好",
        "你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。",
    )
]

model_dir = os.path.join(project_dir, "chatglm2_6b")
# tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
# # model = ChatGLMForConditionalGeneration.from_pretrained(model_dir)
config = ChatGLMConfig.from_pretrained(model_dir)
# # config.num_layers = 1
# model = ChatGLMForConditionalGeneration.from_pretrained(model_dir, config=config)



from transformers import AutoTokenizer, AutoModel
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from accelerate import load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model

model_name = '/home/faith/chatglm2-6b'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
max_mem = {0: 10_79532032, 1: 22_79532032, 'cpu': 912_90877440}
# model = load_checkpoint_and_dispatch(
#     model, '/home/faith/chatglm2-6b', device_map="balanced", no_split_module_classes=["GLMBlock"], offload_folder="/home/faith/langchain-ChatGLM/test", max_memory=max_mem
# )


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
model.eval()

# debug this to get input2
# model.chat(tokenizer=tokenizer, query=prompt)

# test chat speed
"""
all_res = []
print("test chat speed for pytorch model, may cost a lot time", )
test_text = "你好, 请用python写一个链表。"
st = time.time()
for i in trange(10):
    responses, history = model.chat(tokenizer=tokenizer, query=test_text)
    all_res.append(responses)
et = time.time()
tokens = tokenizer.encode("".join(all_res), return_tensors="pt")[0]
token_num = len(tokens)
speed = round(token_num / (et - st), 1)
print("speed: {} tokens/s".format(speed))
"""
input_tensors = build_inputs(device, tokenizer, query, history)
# --- prepare data for input1 ---
input_ids1 = input_tensors["input_ids"]
position_ids1 = input_tensors["position_ids"]
# save input1
pt_input_dict1["input_ids"] = input_ids1[:1].detach().cpu()
pt_input_dict1["position_ids"] = position_ids1[:1].detach()
if args.data_type == "fp16":
    dtype = torch.float16
else:
    dtype = torch.float32
output_dict1 = model.forward(
    input_ids=input_ids1,
    position_ids=position_ids1,
)

# save output1 logists
pt_output_dict1["logits"] = output_dict1["logits"][:1].detach().cpu()
pt_output_dict1["num_layers"] = config.num_layers
past_key_values_1 = output_dict1["past_key_values"]
print("one past_key_shape for input 1 is ", past_key_values_1[0][0].shape)
print("logits for input1 shape is ", output_dict1["logits"].shape)

# --- prepare data for input2 ---
# copy from forward in second time
input_ids2 = torch.tensor([[30910]]).to(device)

# copy from _update_model_kwargs_for_generation in modeling_chatglm.py
new_position_id = position_ids1[..., -1:].clone()
new_position_id += 1
position_ids2 = torch.cat(
    [position_ids1, new_position_id], dim=-1
)
# input_ids2 = torch.cat((input_ids2, input_ids2), dim=0)
# position_ids2 = torch.cat((position_ids2, position_ids2), dim=0)
# attention_mask2 = torch.cat((attention_mask2, attention_mask2), dim=0)
output_dict2 = model.forward(
    input_ids=input_ids2,
    position_ids=position_ids2,
    past_key_values=past_key_values_1,
)
past_key_values_2 = output_dict2["past_key_values"]
print("one past_key_shape for input 2 is ", past_key_values_2[0][0].shape)
print("logits for input2 shape is ", output_dict2["logits"].shape)

# save input2
pt_input_dict2["input_ids"] = input_ids2[:1].detach().cpu()
pt_input_dict2["position_ids"] = position_ids2[:1].detach().cpu()

# save logits2
pt_output_dict2["logits"] = output_dict2["logits"][:1].detach().cpu()
pt_output_dict2["num_layers"] = config.num_layers

for layer_idx in range(model.config.num_layers):
    # --- input key and value ---
    past_key_name = f"past_key_values.{layer_idx}.key"
    past_value_name = f"past_key_values.{layer_idx}.value"
    # --- output key and value ---
    present_key_name = f"present_key_values.{layer_idx}.key"
    present_value_name = f"present_key_values.{layer_idx}.value"

    # save output1 present_key_values 
    present_key = past_key_values_1[layer_idx][0][:,:1].detach().cpu()
    present_value = past_key_values_1[layer_idx][1][:, :1].detach().cpu()
    pt_output_dict1[present_key_name] = present_key
    pt_output_dict1[present_value_name] = present_value

    # save input2 past_key_values
    # input2 past_key_values is same as output1 present_key_values
    pt_input_dict2[past_key_name] = present_key
    pt_input_dict2[past_value_name] = present_value

    # save output2 present_key_values
    present_key2 = past_key_values_2[layer_idx][0][:, :1].detach().cpu()
    present_value2 = past_key_values_2[layer_idx][1][:, :1].detach().cpu()
    pt_output_dict2[present_key_name] = present_key2
    pt_output_dict2[present_value_name] = present_value2


# save input1
input_container1 = torch.jit.script(Container(pt_input_dict1))
input_container1.save(pt_input_path1)

# save output1
output1_container = torch.jit.script(Container(pt_output_dict1))
output1_container.save(pt_output_path1)

# save input2
input2_container = torch.jit.script(Container(pt_input_dict2))
input2_container.save(pt_input_path2)

# save output2
output2_container = torch.jit.script(Container(pt_output_dict2))
output2_container.save(pt_output_path2)