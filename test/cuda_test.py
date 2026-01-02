import torch
print(torch.__version__)          # 应输出 2.3.0
print(torch.version.cuda)         # 应输出 12.1
print(torch.cuda.is_available())  # 有 GPU 且驱动正常 → True
print(torch.rand(3, 3))           # 测试张量创建