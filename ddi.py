import torch
import types
import numpy as np
from functools import partial
import torch.nn.functional as F

# Gaussian High-pass Filter
# Extracts the high-frequency content component φ_c^high from the input feature h
def high_freq_filter(h, radius_ratio=0.3):
    orig_dtype = h.dtype
    h = h.to(torch.float32)

    B, C, H, W = h.shape
    
    # (1) FFT → Frequency-domain transformation (F)
    fft = torch.fft.fft2(h, norm='ortho')
    fft_shift = torch.fft.fftshift(fft)

    # (2) Calculation of the center-to-center distance D
    cy, cx = H // 2, W // 2
    radius = float(min(H, W) * radius_ratio) # r (radius ratio, default: 0.3)

    # (3) Generating a Gaussian high-pass mask
    # M_gauss-high(r) = 1 - exp(-D^2 / (2r^2 + ε))
    y = torch.arange(H, device=h.device).view(-1, 1)
    x = torch.arange(W, device=h.device).view(1, -1)
    dist = (y - cy)**2 + (x - cx)**2
    
    # (4) Applying a filter in the frequency domain
    sigma_sq = (radius**2) + 1e-8 
    mask = 1.0 - torch.exp(-dist / (2 * sigma_sq))
    mask = mask.unsqueeze(0).unsqueeze(0)

    # (5) Inverse FFT → Reconstruction in the spatial domain (F^{-1})
    fft_filtered = fft_shift * mask
    fft_ifftshift = torch.fft.ifftshift(fft_filtered)
    filtered = torch.fft.ifft2(fft_ifftshift, norm='ortho')

    # Final high-frequency feature φ_c^high
    return filtered.real.to(orig_dtype)

# Set all 50 timesteps used by the DDIM sampler for actual sampling as targets for DDI
def make_content_injection_schedule(ddim_timesteps, start_idx=0, end_idx=50):
    return ddim_timesteps[start_idx : end_idx]


# Inserting DDI into the ResBlock of the U-Net decoder
def patch_decoder_resblocks_h_and_cnt_hf(unet, schedule, residuals_all, ratio=0.3):
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
        
        # (1) Basic stylized forward (φ_cs generation)
        if out_layers_injected is not None:
            move_feat_maps_to_device(out_layers_injected, x.device)
            
        out_stylized = orig_forward(x, emb, out_layers_injected)
        
        if out_layers_injected is not None: 
            move_feat_maps_to_cpu(out_layers_injected)
        
        t = getattr(self, "ri_timestep", None)
        # content feature key (φ_c)
        key_h = f"output_block_{self.block_id}_cnt_h"

        out_res = out_stylized
        # (2) Perform injection only at the timesteps specified in the schedule
        if t in schedule:
            idx = int(np.where(schedule == t)[0][0])
            
            # content feature φ_c
            h_cnt = residuals_all[idx].get(key_h, None)
            h_cnt = h_cnt.to(out_stylized.device)
            
            
            if h_cnt is not None:
                # If ratio=0, only the default residual is used without injection
                if ratio == 0:
                    # φ_cs + Δφ_cs
                    out_res = self.out_skip + self.out_h
                else:
                    # (3) high-frequency extraction → φ_c^high
                    h_cnt_hf = high_freq_filter(h_cnt, radius_ratio=ratio)
                    
                    # (4) Discrepancy measurement based on cosine similarity
                    # cos(φ_cs, φ_c)
                    cos_sim = F.cosine_similarity(
                        self.out_h.to(torch.float32).flatten(1), 
                        h_cnt.to(torch.float32).flatten(1), 
                        dim=1
                    )
                    
                    # (5) discrepancy weight ω = 1 - cos(φ_cs, φ_c)
                    diff_weight = (1.0 - cos_sim).view(-1, 1, 1, 1).to(h_cnt.dtype)
                    
                    # (6) Final DDI expression
                    # φ̂_cs = φ_cs + Δφ_cs + ω · φ_c^high
                    out_res = self.out_skip + self.out_h + (h_cnt_hf * diff_weight)
                    
                    del h_cnt_hf

                
            del h_cnt
        return out_res

    # Apply DDI to specific blocks (6–11) of the decoder
    for block_id in range(6, 12):
        if block_id >= len(unet.output_blocks):
            break
        for module in reversed(unet.output_blocks[block_id]):
            if module.__class__.__name__.endswith("ResBlock"):
                module.block_id = block_id

                orig_forward = module._forward
                
                # Replace with a new forward (insert DDI)
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
