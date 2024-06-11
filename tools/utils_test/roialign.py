from torchvision.ops import RoIAlign
import torch

input_map = torch.rand(1,1,8,8)

box =  torch.tensor([[0.0,0.375,0.875,0.625], [0.0,0.375,0.875,0.625]]) 

pooler = RoIAlign(output_size=7, sampling_ratio=2, spatial_scale=1/16)

print(box.shape)

output = pooler(input_map,[box])

print(output.shape)


