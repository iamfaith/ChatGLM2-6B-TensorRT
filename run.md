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

## no attention_mask

Linux, CUDA 11.8  use this
pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html


pip install transformers==4.30.2

export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64/:/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/nvidia/cudnn/lib:/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/torch/lib::/home/faith/miniconda3/envs/py3.10/lib/python3.8/site-packages/tensorrt

con py3.10
python /mnt/ChatGLM2-6B-TensorRT/onnx_export/export2onnx_v2.py



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

use this:
docker run --gpus all \
	-it --name trt \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v ${PWD}:/workspace/ \
	nvcr.io/nvidia/pytorch:23.04-py3

python -m pip install colored transformers sentencepiece onnxruntime

rm -rf kernel/build/ kernel/ckernel.cpython-38-x86_64-linux-gnu.so 
python -m pip install --no-cache-dir -e kernel

- 一个可能的解决方案是**使用pip install --force-reinstall**命令，它可以强制重新安装一个或多个已存在的包，包括重新编译源代码。¹ 例如：
    - `pip install --force-reinstall -e kernel/`
- 另一个可能的解决方案是**使用pip install --no-cache-dir**命令，它可以禁用缓存，从而避免使用已编译的版本。² 例如：
    - `pip install --no-cache-dir -e kernel`
- 还有一个可能的解决方案是**使用pip install --ignore-installed**命令，它可以忽略已安装的包，并重新安装它们。³ 例如：
    - `pip install --ignore-installed kernel/`


Source: Conversation with Bing, 7/14/2023
(1) pip - Python: how to edit an installed package? - Stack Overflow. https://stackoverflow.com/questions/23075397/python-how-to-edit-an-installed-package.
(2) pip install - pip documentation v23.1.2. https://pip.pypa.io/en/stable/cli/pip_install.html.
(3) How to reinstall a pip package even if it exists - Stack Overflow. https://stackoverflow.com/questions/53065940/how-to-reinstall-a-pip-package-even-if-it-exists.


<!-- 32.5 min every time -->
python tensorrt_export/onnx2trt_with_cache.py > trt_with_past.log 2>&1 

检查数据精度，验证TensorRT文件输出结果和pytorch是否一样
python tensorrt_export/trt_check_no_past.py 

python tensorrt_export/trt_check_with_past.py 


trt forward time:
0.11942362785339355
0.029341936111450195
0.029618263244628906
0.026393651962280273
0.025846004486083984
0.024868488311767578
0.024997234344482422
0.025296688079833984
0.02353191375732422
0.02351546287536621


torch forward time:
0.027190685272216797
0.02532029151916504
0.02522873878479004
0.025246620178222656
0.025194168090820312
0.025243520736694336
0.025290250778198242

call stack：
forward (\home\faith\.cache\huggingface\modules\transformers_modules\chatglm-6b\modeling_chatglm.py:1187)
_call_impl (\home\faith\miniconda3\envs\torch_cuda_11.3\lib\python3.8\site-packages\torch\nn\modules\module.py:1501)
greedy_search (\home\faith\miniconda3\envs\torch_cuda_11.3\lib\python3.8\site-packages\transformers\generation\utils.py:2201)
generate (\home\faith\miniconda3\envs\torch_cuda_11.3\lib\python3.8\site-packages\transformers\generation\utils.py:1406)
decorate_context (\home\faith\miniconda3\envs\torch_cuda_11.3\lib\python3.8\site-packages\torch\utils\_contextlib.py:115)
main (\home\faith\langchain-ChatGLM\test\module\benchmark_chatglm.py:123)
<module> (\home\faith\langchain-ChatGLM\test\module\benchmark_chatglm.py:176)
_run_code (\home\faith\miniconda3\envs\torch_cuda_11.3\lib\python3.8\runpy.py:87)
_run_module_as_main (\home\faith\miniconda3\envs\torch_cuda_11.3\lib\python3.8\runpy.py:194)


time benchmark
'/home/faith/miniconda3/envs/torch_cuda_11.3/lib/python3.8/site-packages/transformers/generation/utils.py:2204'

 /home/faith/miniconda3/envs/py3.10/bin/python /mnt/ChatGLM2-6B-TensorRT/onnx_export/run_onnx_cpu.py
 
input number 58
output number 57
=================input names=================
['input_ids', 'position_ids', 'past_key_values.0.key', 'past_key_values.0.value', 'past_key_values.1.key', 'past_key_values.1.value', 'past_key_values.2.key', 'past_key_values.2.value', 'past_key_values.3.key', 'past_key_values.3.value', 'past_key_values.4.key', 'past_key_values.4.value', 'past_key_values.5.key', 'past_key_values.5.value', 'past_key_values.6.key', 'past_key_values.6.value', 'past_key_values.7.key', 'past_key_values.7.value', 'past_key_values.8.key', 'past_key_values.8.value', 'past_key_values.9.key', 'past_key_values.9.value', 'past_key_values.10.key', 'past_key_values.10.value', 'past_key_values.11.key', 'past_key_values.11.value', 'past_key_values.12.key', 'past_key_values.12.value', 'past_key_values.13.key', 'past_key_values.13.value', 'past_key_values.14.key', 'past_key_values.14.value', 'past_key_values.15.key', 'past_key_values.15.value', 'past_key_values.16.key', 'past_key_values.16.value', 'past_key_values.17.key', 'past_key_values.17.value', 'past_key_values.18.key', 'past_key_values.18.value', 'past_key_values.19.key', 'past_key_values.19.value', 'past_key_values.20.key', 'past_key_values.20.value', 'past_key_values.21.key', 'past_key_values.21.value', 'past_key_values.22.key', 'past_key_values.22.value', 'past_key_values.23.key', 'past_key_values.23.value', 'past_key_values.24.key', 'past_key_values.24.value', 'past_key_values.25.key', 'past_key_values.25.value', 'past_key_values.26.key', 'past_key_values.26.value', 'past_key_values.27.key', 'past_key_values.27.value']
=================output names=================
['logits', 'present_key_values.0.key', 'present_key_values.0.value', 'present_key_values.1.key', 'present_key_values.1.value', 'present_key_values.2.key', 'present_key_values.2.value', 'present_key_values.3.key', 'present_key_values.3.value', 'present_key_values.4.key', 'present_key_values.4.value', 'present_key_values.5.key', 'present_key_values.5.value', 'present_key_values.6.key', 'present_key_values.6.value', 'present_key_values.7.key', 'present_key_values.7.value', 'present_key_values.8.key', 'present_key_values.8.value', 'present_key_values.9.key', 'present_key_values.9.value', 'present_key_values.10.key', 'present_key_values.10.value', 'present_key_values.11.key', 'present_key_values.11.value', 'present_key_values.12.key', 'present_key_values.12.value', 'present_key_values.13.key', 'present_key_values.13.value', 'present_key_values.14.key', 'present_key_values.14.value', 'present_key_values.15.key', 'present_key_values.15.value', 'present_key_values.16.key', 'present_key_values.16.value', 'present_key_values.17.key', 'present_key_values.17.value', 'present_key_values.18.key', 'present_key_values.18.value', 'present_key_values.19.key', 'present_key_values.19.value', 'present_key_values.20.key', 'present_key_values.20.value', 'present_key_values.21.key', 'present_key_values.21.value', 'present_key_values.22.key', 'present_key_values.22.value', 'present_key_values.23.key', 'present_key_values.23.value', 'present_key_values.24.key', 'present_key_values.24.value', 'present_key_values.25.key', 'present_key_values.25.value', 'present_key_values.26.key', 'present_key_values.26.value', 'present_key_values.27.key', 'present_key_values.27.value']


polygraphy inspect model output/onnx_output/chatglm2_6b.onnx

---- 58 Graph Input(s) ----
    {input_ids [dtype=int64, shape=('batch_size', 'sequence')],
     position_ids [dtype=int64, shape=('batch_size', 'sequence')],
     past_key_values.0.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.0.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.1.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.1.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.2.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.2.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.3.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.3.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.4.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.4.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.5.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.5.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.6.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.6.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.7.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.7.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.8.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.8.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.9.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.9.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.10.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.10.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.11.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.11.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.12.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.12.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.13.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.13.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.14.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.14.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.15.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.15.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.16.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.16.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.17.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.17.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.18.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.18.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.19.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.19.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.20.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.20.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.21.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.21.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.22.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.22.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.23.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.23.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.24.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.24.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.25.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.25.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.26.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.26.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.27.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
     past_key_values.27.value [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)]}

    ---- 57 Graph Output(s) ----
    {logits [dtype=float32, shape=('batch_size', 'sequence', 65024)],
     present_key_values.0.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.0.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.1.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.1.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.2.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.2.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.3.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.3.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.4.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.4.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.5.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.5.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.6.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.6.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.7.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.7.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.8.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.8.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.9.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.9.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.10.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.10.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.11.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.11.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.12.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.12.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.13.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.13.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.14.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.14.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.15.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.15.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.16.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.16.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.17.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.17.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.18.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.18.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.19.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.19.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.20.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.20.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.21.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.21.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.22.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.22.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.23.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.23.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.24.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.24.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.25.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.25.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.26.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.26.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.27.key [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)],
     present_key_values.27.value [dtype=float32, shape=('past_sequence + sequence', 'batch_size', 2, 128)]}

    ---- 199 Initializer(s) ----
