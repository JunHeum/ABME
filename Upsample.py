import torch
import torch.nn.functional as F

def upsample_kernel2d(w, device):
    c = w // 2
    kernel = 1 - torch.abs(c - torch.arange(w, dtype=torch.float32, device=device)) / (c + 1)
    kernel = kernel.repeat(w).view(w,-1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w, w)

def Upsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B*C, 1, H, W)
    batch_img = F.pad(batch_img, [0, 1, 0, 1], mode='replicate')
    kernel = upsample_kernel2d(factor * 2 - 1, img.device)
    upsamp_img = F.conv_transpose2d(batch_img, kernel, stride=factor, padding=(factor-1))
    upsamp_img = upsamp_img[:, :, : -1, :-1]
    _, _, H_up, W_up = upsamp_img.shape
    return upsamp_img.view(B, C, H_up, W_up)