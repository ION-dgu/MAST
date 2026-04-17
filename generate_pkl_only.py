import argparse
import os
import torch
import torch.nn.functional as F
from torch import autocast
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from contextlib import nullcontext
import copy
import pickle
import time

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


feat_maps = []


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

SUPPORTED_INFERENCE_HW = {
    (512, 512),
    (512, 768),
    (512, 1024),
    (768, 512),
    (768, 768),
    (768, 1024),
    (1024, 512),
    (1024, 768),
    (1024, 1024),
}


def is_mask_image(fname):
    """
    Skip per-content mask files such as `content_020_mask0.png`.

    The pkl precompute step should only run on real content/style images, not
    on the auxiliary mask PNGs stored alongside them.
    """
    base_name, ext = os.path.splitext(fname.lower())
    return ext == ".png" and "_mask" in base_name


def resolve_runtime_resolution(opt, model_config):
    """
    Resolve the runtime image size `(H, W)` from the model defaults plus CLI
    overrides.

    `generate_pkl_only.py` and `run_ori.py` must use the exact same rule to
    avoid precomputed feature shape mismatches.
    """
    cfg_h, cfg_w = None, None
    if model_config is not None and "inference" in model_config:
        inference_cfg = model_config.inference
        cfg_h = inference_cfg.get("default_h", None)
        cfg_w = inference_cfg.get("default_w", None)
        if (cfg_h is None or cfg_w is None) and "default_hw" in inference_cfg:
            default_hw = inference_cfg.get("default_hw")
            if default_hw is not None and len(default_hw) == 2:
                cfg_h = default_hw[0]
                cfg_w = default_hw[1]

    h = opt.H if opt.H is not None else (cfg_h if cfg_h is not None else 512)
    w = opt.W if opt.W is not None else (cfg_w if cfg_w is not None else 512)
    return int(h), int(w)


def validate_runtime_resolution(h, w, f):
    """
    Validate the supported runtime resolution list and latent downsampling
    factor `f`.
    """
    if (h, w) not in SUPPORTED_INFERENCE_HW:
        raise ValueError(
            f"Unsupported runtime resolution: ({h}, {w}). "
            f"Supported resolutions: {sorted(SUPPORTED_INFERENCE_HW)}"
        )
    if h % f != 0 or w % f != 0:
        raise ValueError(f"H and W must be divisible by f={f}. Received H={h}, W={w}, f={f}")


def get_resolution_scoped_precomputed_dir(base_dir, h, w):
    """
    Resolve the feature-pkl output directory by runtime resolution.

    - `512x512`: keep the legacy layout and use `base_dir` directly.
    - Other supported resolutions: use `{base_dir}/{H}x{W}`.
    """
    if (h, w) == (512, 512):
        return base_dir
    return os.path.join(base_dir, f"{h}x{w}")


def resize_tensor_hw(x, target_h, target_w, mode="bilinear"):
    """
    Resize a 4D tensor `(B, C, H, W)` to `target_h x target_w`.

    Resize is used instead of cropping so the full image layout is preserved.
    """
    if x.ndim != 4:
        raise ValueError(f"resize_tensor_hw only supports 4D tensors. Received shape={tuple(x.shape)}")

    if mode in {"bilinear", "bicubic", "trilinear", "linear"}:
        return F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=False)
    return F.interpolate(x, size=(target_h, target_w), mode=mode)


def load_img(path, target_h=512, target_w=512):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    image = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.
    image = resize_tensor_hw(image, target_h, target_w, mode="bilinear")
    return image


def list_image_paths(folder, recursive=False):
    """
    Collect image file paths from a folder.

    - `recursive=True`: traverse all nested subdirectories (used for styles)
    - `recursive=False`: scan only the current directory
    """
    if not folder or not os.path.isdir(folder):
        return []

    image_paths = []
    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                if is_mask_image(name):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in VALID_IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if is_mask_image(name):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                image_paths.append(path)

    image_paths.sort()
    return image_paths

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = None, help='Content image folder path')
    parser.add_argument('--sty', default = None, help='Style image folder path')
    parser.add_argument('--ddim_inv_steps', type=int, default=50)
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    # If omitted on the CLI, use the inference defaults from `model_config`.
    parser.add_argument('--H', type=int, default=None)
    parser.add_argument('--W', type=int, default=None)
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml')
    parser.add_argument('--precomputed', type=str, default='data/precomputed_feats')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument('--data_root', type=str, default='./data')
    opt = parser.parse_args()

    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_config = OmegaConf.load(opt.model_config)
    # Use the same resolution rule as `run_ori.py` to avoid pkl shape mismatches.
    opt.H, opt.W = resolve_runtime_resolution(opt, model_config)
    validate_runtime_resolution(opt.H, opt.W, opt.f)
    print(f"Runtime resolution for precompute: H={opt.H}, W={opt.W}")

    feat_path_root = get_resolution_scoped_precomputed_dir(opt.precomputed, opt.H, opt.W)
    os.makedirs(feat_path_root, exist_ok=True)

    model = load_model_from_config(model_config, opt.ckpt)
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    
    for name, module in unet_model.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            module.gen_pkl = True
            module.base_latent_hw = (opt.H // opt.f, opt.W // opt.f)
            print(f"Set gen_pkl=True for {name}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps
    

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)

    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t
    
    global feat_maps
    # Rebuild the per-image buffer with the current DDIM schedule length.
    def reset_feat_maps_buffer():
        global feat_maps
        feat_maps = [{'config': {'T': 1.5}} for _ in range(len(time_range))]

    reset_feat_maps_buffer()

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map_z(xt, 'z_enc', i)
    
    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q.detach().cpu()
                    k = block[1].transformer_blocks[0].attn1.k.detach().cpu()
                    v = block[1].transformer_blocks[0].attn1.v.detach().cpu()
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1
            
    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map_z(xt, name, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][name] = xt.detach().cpu()

    def residual_injection_callback(pred_x0, xt, t):
        # Save feature maps captured during inversion.
        save_feature_maps_callback(t)
        save_feature_map_z(xt, 'z_enc', t)

        t_int = int(t)
        if t_int not in residuals_all:
            residuals_all[t_int] = {}

        for block_id in range(6, 12):
            if block_id >= len(unet_model.output_blocks):
                break

            for module in reversed(unet_model.output_blocks[block_id]):
                if module.__class__.__name__.endswith("ResBlock"):               

                    if hasattr(module, 'out_h') and module.out_h is not None:
                        h = module.out_h.detach().cpu()
                        key_h = f"output_block_{block_id}_cnt_h"
                        save_feature_map(h, f"{key_h}", t_int)
                        print(f"[Callback] t={t_int}, saved {key_h}")
                 

                    break  # Only process the final ResBlock.

                    
    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    # ==========================
    # Resolve the content/style folders used for precomputation.
    # ==========================
    # If `--cnt/--sty` is provided, use those folders first.
    # Otherwise, fall back to the default folders under `data_root`.
    cnt_folder = opt.cnt if opt.cnt is not None else os.path.join(opt.data_root, "cnt")
    sty_folder = opt.sty if opt.sty is not None else os.path.join(opt.data_root, "sty")

    # Content images usually live directly under `cnt`, so the default scan is
    # non-recursive. Styles are scanned recursively to support layouts such as
    # `char/`, `back/`, or other nested style groups.
    cnt_image_paths = list_image_paths(cnt_folder, recursive=False)
    sty_image_paths = list_image_paths(sty_folder, recursive=True)

    if not cnt_image_paths:
        print(f"[Info] No content images found in: {cnt_folder}")
    if not sty_image_paths:
        print(f"[Info] No style images found in: {sty_folder}")

    # Pkl file names are built from basenames, so different files with the same
    # basename would overwrite each other. `run_ori.py` also resolves by
    # basename, so it is safer to reject collisions explicitly here.
    seen_output_to_source = {}

    # ===== STYLE FEATURE ( *_sty.pkl ) =====
    for sty_path in sty_image_paths:
        sty_name = os.path.basename(sty_path)
        sty_feat_name = os.path.join(
            feat_path_root, os.path.splitext(sty_name)[0] + '_sty.pkl'
        )

        prev_src = seen_output_to_source.get(sty_feat_name)
        if prev_src is not None and os.path.abspath(prev_src) != os.path.abspath(sty_path):
            raise ValueError(
                f"Style basename collision detected for the same pkl path:\n"
                f" - {prev_src}\n - {sty_path}\n"
                f" -> {sty_feat_name}"
            )
        seen_output_to_source[sty_feat_name] = sty_path

        if os.path.isfile(sty_feat_name):
            print(f"Precomputed style feature exists: {sty_feat_name}")
            continue

        # Reset the feature buffer for each image so stale keys do not leak in.
        reset_feat_maps_buffer()
        init_sty = load_img(sty_path, target_h=opt.H, target_w=opt.W).to(device)
        init_sty_latent = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
        sty_z_enc, _ = sampler.encode_ddim(
            init_sty_latent.clone(),
            num_steps=ddim_inversion_steps,
            unconditional_conditioning=uc,
            end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
            callback_ddim_timesteps=save_feature_timesteps,
            img_callback=ddim_sampler_callback
        )
        with open(sty_feat_name, 'wb') as f:
            pickle.dump(copy.deepcopy(feat_maps), f)
        print(f"Saved style feature: {sty_feat_name}")

    for cnt_path in cnt_image_paths:
        cnt_name = os.path.basename(cnt_path)
        cnt_feat_name = os.path.join(
            feat_path_root, os.path.splitext(cnt_name)[0] + '_cnt.pkl'
        )

        prev_src = seen_output_to_source.get(cnt_feat_name)
        if prev_src is not None and os.path.abspath(prev_src) != os.path.abspath(cnt_path):
            raise ValueError(
                f"Content basename collision detected for the same pkl path:\n"
                f" - {prev_src}\n - {cnt_path}\n"
                f" -> {cnt_feat_name}"
            )
        seen_output_to_source[cnt_feat_name] = cnt_path

        if os.path.isfile(cnt_feat_name):
            print(f"Precomputed content feature exists: {cnt_feat_name}")
            continue

        reset_feat_maps_buffer()
        init_cnt = load_img(cnt_path, target_h=opt.H, target_w=opt.W).to(device)
        init_cnt_latent = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
        residuals_all = {}
        cnt_z_enc, _ = sampler.encode_ddim(
            init_cnt_latent.clone(),
            num_steps=ddim_inversion_steps,
            unconditional_conditioning=uc,
            end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
            callback_ddim_timesteps=save_feature_timesteps,
            img_callback=residual_injection_callback
        )
        with open(cnt_feat_name, 'wb') as f:
            pickle.dump(copy.deepcopy(feat_maps), f)
        print(f"Saved content feature: {cnt_feat_name}")

if __name__ == "__main__":
    main()
