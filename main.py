import torch

if (torch.backends.mps.is_available()):
    device = torch.device('mps')

# print(torch.backends.mps.is_built())
#
# device = torch.device('mps' if torch.backends.mps.is_available())

print(device)

