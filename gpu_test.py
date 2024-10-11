import sys
print(sys.executable)  # 检查 Python 解释器的路径

import torch
print(torch.__version__)
print(torch.__file__)  # 检查 PyTorch 的安装路径

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    print(f"当前使用的设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA 不可用")
