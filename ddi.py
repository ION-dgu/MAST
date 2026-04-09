import torch
import types
import numpy as np
from functools import partial
import torch.nn.functional as F

# before filter
def high_freq_filter(h, radius_ratio=0.5):
    orig_dtype = h.dtype
    h = h.to(torch.float32)

    B, C, H, W = h.shape
    fft = torch.fft.fft2(h, norm='ortho')
    fft_shift = torch.fft.fftshift(fft)

    cy, cx = H // 2, W // 2
    radius = float(min(H, W) * radius_ratio) 

    y = torch.arange(H, device=h.device).view(-1, 1)
    x = torch.arange(W, device=h.device).view(1, -1)
    dist = (y - cy)**2 + (x - cx)**2
    
    sigma_sq = (radius**2) + 1e-8 
    mask = 1.0 - torch.exp(-dist / (2 * sigma_sq))
    mask = mask.unsqueeze(0).unsqueeze(0)

    fft_filtered = fft_shift * mask
    fft_ifftshift = torch.fft.ifftshift(fft_filtered)
    filtered = torch.fft.ifft2(fft_ifftshift, norm='ortho')

    return filtered.real.to(orig_dtype)

def make_content_injection_schedule(ddim_timesteps, start_idx=0, end_idx=50):
    return ddim_timesteps[start_idx : end_idx]


def patch_decoder_resblocks_h_and_cnt_hf(unet, schedule, residuals_all, ratio=0.5):
    def move_feat_maps_to_device(feat_maps, device):
        for i, f in enumerate(feat_maps):
            if isinstance(f, dict):
                for k, v in f.items():
                    if torch.is_tensor(v):
                        f[k] = v.to(device)
            elif torch.is_tensor(f):
                feat_maps[i] = f.to(device)
        return feat_maps
    def move_feat_maps_to_cpu(feat_maps):
        for i, f in enumerate(feat_maps):
            if isinstance(f, dict):
                for k, v in f.items():
                    if torch.is_tensor(v):
                        f[k] = v.cpu()
            elif torch.is_tensor(f):
                feat_maps[i] = f.cpu()
        return feat_maps

    @torch.no_grad()
    def wrapped_forward(self, x, emb, out_layers_injected=None, *, orig_forward, schedule, residuals_all, ratio):
        
        if out_layers_injected is not None:
            move_feat_maps_to_device(out_layers_injected, x.device)
        out_stylized = orig_forward(x, emb, out_layers_injected)
        if out_layers_injected is not None: 
            move_feat_maps_to_cpu(out_layers_injected)
        
        t = getattr(self, "ri_timestep", None)
        key_h = f"output_block_{self.block_id}_cnt_h"

        out_res = out_stylized
        if t in schedule:
            idx = int(np.where(schedule == t)[0][0])
            h_cnt = residuals_all[idx].get(key_h, None)
            h_cnt = h_cnt.to(out_stylized.device)
            
            
            if h_cnt is not None:
                if ratio == 0:
                    
                    out_res = self.out_skip + self.out_h
                else:
                    h_cnt_hf = high_freq_filter(h_cnt, radius_ratio=ratio)
                    
                    cos_sim = F.cosine_similarity(
                        self.out_h.to(torch.float32).flatten(1), 
                        h_cnt.to(torch.float32).flatten(1), 
                        dim=1
                    )
                    
                    diff_weight = (1.0 - cos_sim).view(-1, 1, 1, 1).to(h_cnt.dtype)
                    
                    out_res = self.out_skip + self.out_h + (h_cnt_hf * diff_weight)
                    
                    del h_cnt_hf

                
            del h_cnt
        return out_res

    for block_id in range(6, 12):
        if block_id >= len(unet.output_blocks):
            break
        for module in reversed(unet.output_blocks[block_id]):
            if module.__class__.__name__.endswith("ResBlock"):
                module.block_id = block_id
                orig_forward = module._forward
                module._forward = types.MethodType(
                    partial(
                        wrapped_forward,
                        orig_forward=orig_forward,
                        schedule=schedule,
                        residuals_all=residuals_all,
                        ratio=ratio
                    ),
                    module
                )
                break