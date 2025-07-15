
from torchvision.transforms import v2 as transforms_v2
import torch

print(torch.cuda.is_available())
print(torch.cuda.memory_allocated(device='cuda'))
print(torch.cuda.memory_reserved(device='cuda'))


print(torch.cuda.get_device_properties(device='cuda').total_memory/1024**3)