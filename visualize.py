import torch
import numpy as np
from matplotlib import pyplot as plt

ckpt = 'log/tensorf_ship_VM/tensorf_ship_VM.th'
model = torch.load(ckpt)

# print(model['state_dict']['density_plane.1'][0][[0]].cpu().shape)

for i in range(model['state_dict']['app_plane.2'][0].shape[0]):
    img = model['state_dict']['app_plane.2'][0][[i]].cpu()
    img = torch.permute(img, (1, 2, 0)).numpy()
    img = np.flip(img, (0, 1))
    plt.imshow(img, interpolation='nearest')
    plt.show(block=False)
    plt.pause(1)
    plt.close()