您好，这是必应。根据我从网络上搜索到的信息，这些警告是由于您的模型中存在一些数据依赖的控制流（如if或for循环），而torch.onnx.export函数只能追踪张量上的操作，而不能记录Python值的数据流¹。这意味着导出的模型可能不适用于其他输入²。您可以尝试以下方法来处理这些警告：

- 确保您导出模型时传入的参数不会改变（如None值或配置信息），并且在后续使用模型时也保持一致¹³。
- 使用torch.jit.script而不是torch.onnx.export来导出模型，因为torch.jit.script可以支持数据依赖的控制流¹²。
- 忽略这些警告，如果您确定导出的模型在您的用例下能够正常工作²³。

希望这些方法能够帮助您成功导出模型。如果还有其他问题，请随时与我联系。😊

Source: Conversation with Bing, 7/13/2023
(1) Torch JIT Trace = TracerWarning: Converting a tensor to a .... https://stackoverflow.com/questions/66746307/torch-jit-trace-tracerwarning-converting-a-tensor-to-a-python-boolean-might-c.
(2) TracerWarning: Converting a tensor to a Python boolean .... https://github.com/ultralytics/yolov5/issues/5505.
(3) Converting a tensor to a Python boolean might cause the trace .... https://github.com/huggingface/transformers/issues/17773.



PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /home/faith/miniconda3/envs/torch_cuda_11.3/bin/python /mnt/ChatGLM2-6B-TensorRT/onnx_export/export2onnx.py



PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python onnx_export/export_compare_data.py --data_type=fp16


PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python onnx_export/run_onnx_cuda.py



# Install polygraphy
pip install colored polygraphy tensorrt --extra-index-url https://pypi.ngc.nvidia.com

# Install optional dependencies
pip install onnx onnxruntime tensorflow --extra-index-url https://pypi.ngc.nvidia.com



run in local:
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64/:/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/nvidia/cudnn/lib:/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/torch/lib::/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/tensorrt

con py3.10:
python onnx_export/export_compare_data.py --data_type=fp16

compare data:
/home/faith/miniconda3/envs/py3.10/bin/python /mnt/ChatGLM2-6B-TensorRT/onnx_export/run_onnx_cpu.py



# run tensorrt


not used this:
docker run --rm -it --gpus all -v `pwd`:/lyraChatGLM nvcr.io/nvidia/pytorch:23.02-py3

# 临时进入容器（退出后容器自动关闭）

转换的trt和运行的要保持一致


<!-- 8.6.1.2 版本 -->
docker run --gpus all \
	-it --rm \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v ${PWD}:/workspace/ \
	nvcr.io/nvidia/pytorch:23.04-py3

python -m pip install colored
python tensorrt_export/onnx2trt_with_cache.py > trt_with_past.log 2>&1 

检查数据精度，验证TensorRT文件输出结果和pytorch是否一样
python tensorrt_export/trt_check_with_past.py 
python tensorrt_export/trt_check_no_past.py 




input number 58
output number 57
=================input names=================
['input_ids', 'position_ids', 'past_key_values.0.key', 'past_key_values.0.value', 'past_key_values.1.key', 'past_key_values.1.value', 'past_key_values.2.key', 'past_key_values.2.value', 'past_key_values.3.key', 'past_key_values.3.value', 'past_key_values.4.key', 'past_key_values.4.value', 'past_key_values.5.key', 'past_key_values.5.value', 'past_key_values.6.key', 'past_key_values.6.value', 'past_key_values.7.key', 'past_key_values.7.value', 'past_key_values.8.key', 'past_key_values.8.value', 'past_key_values.9.key', 'past_key_values.9.value', 'past_key_values.10.key', 'past_key_values.10.value', 'past_key_values.11.key', 'past_key_values.11.value', 'past_key_values.12.key', 'past_key_values.12.value', 'past_key_values.13.key', 'past_key_values.13.value', 'past_key_values.14.key', 'past_key_values.14.value', 'past_key_values.15.key', 'past_key_values.15.value', 'past_key_values.16.key', 'past_key_values.16.value', 'past_key_values.17.key', 'past_key_values.17.value', 'past_key_values.18.key', 'past_key_values.18.value', 'past_key_values.19.key', 'past_key_values.19.value', 'past_key_values.20.key', 'past_key_values.20.value', 'past_key_values.21.key', 'past_key_values.21.value', 'past_key_values.22.key', 'past_key_values.22.value', 'past_key_values.23.key', 'past_key_values.23.value', 'past_key_values.24.key', 'past_key_values.24.value', 'past_key_values.25.key', 'past_key_values.25.value', 'past_key_values.26.key', 'past_key_values.26.value', 'past_key_values.27.key', 'past_key_values.27.value']
=================output names=================
['logits', 'present_key_values.0.key', 'present_key_values.0.value', 'present_key_values.1.key', 'present_key_values.1.value', 'present_key_values.2.key', 'present_key_values.2.value', 'present_key_values.3.key', 'present_key_values.3.value', 'present_key_values.4.key', 'present_key_values.4.value', 'present_key_values.5.key', 'present_key_values.5.value', 'present_key_values.6.key', 'present_key_values.6.value', 'present_key_values.7.key', 'present_key_values.7.value', 'present_key_values.8.key', 'present_key_values.8.value', 'present_key_values.9.key', 'present_key_values.9.value', 'present_key_values.10.key', 'present_key_values.10.value', 'present_key_values.11.key', 'present_key_values.11.value', 'present_key_values.12.key', 'present_key_values.12.value', 'present_key_values.13.key', 'present_key_values.13.value', 'present_key_values.14.key', 'present_key_values.14.value', 'present_key_values.15.key', 'present_key_values.15.value', 'present_key_values.16.key', 'present_key_values.16.value', 'present_key_values.17.key', 'present_key_values.17.value', 'present_key_values.18.key', 'present_key_values.18.value', 'present_key_values.19.key', 'present_key_values.19.value', 'present_key_values.20.key', 'present_key_values.20.value', 'present_key_values.21.key', 'present_key_values.21.value', 'present_key_values.22.key', 'present_key_values.22.value', 'present_key_values.23.key', 'present_key_values.23.value', 'present_key_values.24.key', 'present_key_values.24.value', 'present_key_values.25.key', 'present_key_values.25.value', 'present_key_values.26.key', 'present_key_values.26.value', 'present_key_values.27.key', 'present_key_values.27.value']