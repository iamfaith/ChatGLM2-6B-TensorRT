import torch
import onnxruntime as ort
import numpy as np
import os
from colored import fg, stylize
# os.environ["TRANSFORMERS_OFFLINE"] = "0"

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
# model_dir = os.path.join(project_dir, "chatglm_6b")
onnx_path_with_cache = os.path.join(output_dir, "onnx_output", "chatglm2_6b.onnx")
onnx_path_no_cache = os.path.join(output_dir, "onnx_output_no_cache", "chatglm2_6b.onnx")


def compare_value(pre_numpy: np.array, true_numpy: np.array):
    assert pre_numpy.shape == true_numpy.shape
    diff = np.abs(pre_numpy - true_numpy).max()
    if diff > 1e-3:
        print(stylize(f"diff: {diff} is_pass: failed", fg("red")))
    else:
        print(stylize(f"diff: {diff} is_pass: OK", fg("green")))
    return diff


def run_cuda_onnx_inference(onnx_path, input_path: str, output_path):
    providers = [("CUDAExecutionProvider", {'enable_cuda_graph': False})]
    sess_options = ort.SessionOptions()

    session = ort.InferenceSession(
        onnx_path, sess_options=sess_options, providers=providers
    )
    print(session.get_providers())

    # cuda device id
    device_id = 0
    
    input_dict = torch.jit.load(input_path)
    print(input_dict)
    output_dict = torch.jit.load(output_path)
    input_ids = input_dict.input_ids.data.cpu().numpy().astype(np.int64)
    position_ids = input_dict.position_ids.data.cpu().numpy().astype(np.int64)
    attention_mask = input_dict.attention_mask.data.cpu().numpy()
    logits = output_dict.logits.data.cpu().numpy()
    key = "present_key_values.0.key"
    one_present_key = getattr(output_dict, key).data.cpu().numpy()
    num_layers = getattr(output_dict, "num_layers")
    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input(
        "input_ids",
        ort.OrtValue.ortvalue_from_numpy(input_ids, "cuda", device_id=device_id)
    )
    io_binding.bind_ortvalue_input(
        "position_ids",
        ort.OrtValue.ortvalue_from_numpy(position_ids, "cuda", device_id=device_id)
    )
    io_binding.bind_ortvalue_input(
        "attention_mask",
        ort.OrtValue.ortvalue_from_numpy(attention_mask, "cuda", device_id=device_id)
    )
    for layer_idx in range(num_layers):
        input_names = [
            f"past_key_values.{layer_idx}.key",
            f"past_key_values.{layer_idx}.value"
        ]
        # inputs[input_names[0]] = past_key_values
        # inputs[input_names[1]] = past_key_values
        for name in input_names:
            try:
                past_key_values = getattr(input_dict, name).data.cpu().numpy()
                io_binding.bind_ortvalue_input(
                    name=name,
                    ortvalue=ort.OrtValue.ortvalue_from_numpy(
                        past_key_values, "cuda", device_id=device_id
                    )
                )
            except Exception:
                past_key_values = np.zeros(
                    [1, input_ids.shape[1], 32, 128],
                    dtype=one_present_key.dtype
                )
            # io_binding.bind_cpu_input(
            #     name,
            #     past_key_values
            # )

        output_name = [
            f"present_key_values.{layer_idx}.key",
            f"present_key_values.{layer_idx}.value"
        ]
        for name in output_name:
            output_value = np.zeros_like(
                one_present_key,
                dtype=one_present_key.dtype
            )
            io_binding.bind_ortvalue_output(
                name=name,
                ortvalue=ort.OrtValue.ortvalue_from_numpy(
                    output_value, "cuda", device_id=device_id
                )
            )
    logits_numpy = np.zeros_like(logits, dtype=logits.dtype)
    io_binding.bind_ortvalue_output(
        name="logits",
        ortvalue=ort.OrtValue.ortvalue_from_numpy(
            logits_numpy
        )
    )

    # print(inputs)
    session.run_with_iobinding(io_binding)
    # compile logists
    print('=' * 20)
    print("compare logits")
    pred_outputs = io_binding.copy_outputs_to_cpu()
    compare_value(pred_outputs[-1], logits)

    # compile present_key_values
    for i in range(num_layers):
        key_name = f"present_key_values.{i}.key"
        value_name = f"present_key_values.{i}.value"
        print('=' * 20)
        print(f"compare {key_name}")
        # key_numpy = [key_name]
        key_true = getattr(output_dict, key_name).data.cpu().numpy()
        key_pred = pred_outputs[i * 2]
        compare_value(key_pred, key_true)
        print('=' * 20)
        print(f"compare {value_name}")
        value_pred = pred_outputs[i * 2 + 1]
        value_true = getattr(output_dict, value_name).data.cpu().numpy()
        compare_value(value_pred, value_true)


if __name__ == "__main__":
    input_path1 = os.path.join(output_dir, "pt_input1.pt")
    output_path1 = os.path.join(output_dir, "pt_output1.pt")
    run_cuda_onnx_inference(onnx_path_with_cache, input_path1, output_path1)
    print("\n")
    input_path2 = os.path.join(output_dir, "pt_input2.pt")
    output_path2 = os.path.join(output_dir, "pt_output2.pt")
    run_cuda_onnx_inference(onnx_path_no_cache, input_path2, output_path2)


