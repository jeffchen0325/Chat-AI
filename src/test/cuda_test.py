import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(torch.version.cuda)  # 应输出 12.1
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Supported archs: {torch.cuda.get_arch_list()}")
print(torch.rand(3, 3))           # 测试张量创建