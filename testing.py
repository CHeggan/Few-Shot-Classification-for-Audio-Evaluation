import torch
import time
from dataset_.dataset_utils import enforce_length



import torch

def split_tensor_and_enforce(tensor, split_size):
    splits = list(torch.split(tensor, split_size, dim=0))

    last_split = enforce_length(splits[-1].unsqueeze(0), split_size).squeeze()

    splits = splits[:-1]
    splits.append(last_split)

    splits = torch.stack(splits, dim=0)
    return splits


start = time.time()
# Example usage
tensor = torch.rand((10, 8))  # Shape: (21,)
split_size = 48000
tensor = torch.flatten(tensor)

splits = split_tensor_and_enforce(tensor, split_size)
print(splits.shape)
# Print the split tensors
for i, split in enumerate(splits):
    print(f"Split {i + 1}: {split.shape}")

end = time.time()

print(end-start)