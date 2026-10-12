import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU可用:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))