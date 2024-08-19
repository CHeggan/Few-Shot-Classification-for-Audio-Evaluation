from pase.models.frontend import wf_builder
pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
pase.load_pretrained('pase.ckpt', load_last=True, verbose=True)

import torch
print(torch.cuda.is_available())

pase = pase.cuda()



# Now we can forward waveforms as Torch tensors
import torch
x = torch.randn(10, 1, 160000).cuda() # example with random noise to check shape
# y size will be (1, 256, 625), which are 625 frames of 256 dims each
with torch.no_grad():
    y = pase(x)


print(y.shape)