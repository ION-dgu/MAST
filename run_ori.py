import copy
import gc
import os
import pickle
import sys
import time

import argparse
import numpy as np
import psutil
import torch
import torch.nn.functional as F

from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything


from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.modules.attention import CrossAttention
try:
    from ACMMM_MAST.ddi import patch_decoder_resblocks_h_and_cnt_hf, make_content_injection_schedule
except ImportError:
    from ddi import patch_decoder_resblocks_h_and_cnt_hf, make_content_injection_schedule

process = psutil.Process()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "data_vis")
DEFAULT_PRECOMPUTED_DIR = os.path.join(SCRIPT_DIR, "precomputed_feats_k")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_dk")
DEFAULT_MODEL_CONFIG = os.path.join(SCRIPT_DIR, "models/ldm/stable-diffusion-v1/v1-inference.yaml")
DEFAULT_CKPT = os.path.join(SCRIPT_DIR, "models/ldm/stable-diffusion-v1/model.ckpt")
DEFAULT_META_FILE = "nstyle_meta.txt"

# Keep the allowed inference resolutions explicit.
# Expanding the official benchmark set should only require editing this table.
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


def resolve_runtime_resolution(opt, model_config):
    """
    Resolve the runtime image size from model defaults plus CLI overrides.

    Priority:
    1. `--H` / `--W`
    2. `model_config.inference.default_h/default_w`
    3. `512 x 512`
    """
    cfg_h, cfg_w = None, None

    if model_config is not None and "inference" in model_config:
        inference_cfg = model_config.inference
        cfg_h = inference_cfg.get("default_h", None)
        cfg_w = inference_cfg.get("default_w", None)

        # Keep compatibility with configs that still expose `default_hw: [H, W]`.
        if (cfg_h is None or cfg_w is None) and "default_hw" in inference_cfg:
            default_hw = inference_cfg.get("default_hw")
            if default_hw is not None and len(default_hw) == 2:
                cfg_h = default_hw[0]
                cfg_w = default_hw[1]

    h = opt.H if opt.H is not None else (cfg_h if cfg_h is not None else 512)
    w = opt.W if opt.W is not None else (cfg_w if cfg_w is not None else 512)

    h = int(h)
    w = int(w)
    return h, w


def validate_runtime_resolution(h, w, f):
    """
    Validate that the runtime resolution is officially supported and matches the
    latent downsampling factor.
    """
    if (h, w) not in SUPPORTED_INFERENCE_HW:
        raise ValueError(
            f"Unsupported runtime resolution: ({h}, {w}). "
            f"Supported resolutions: {sorted(SUPPORTED_INFERENCE_HW)}"
        )

    if h % f != 0 or w % f != 0:
        raise ValueError(
            f"H and W must be divisible by f={f}. Received H={h}, W={w}."
        )


def get_resolution_scoped_precomputed_dir(base_dir, h, w):
    """
    Scope precomputed features by resolution.

    - `512x512` keeps the legacy directory layout.
    - Other supported resolutions use `{base_dir}/{H}x{W}`.
    """
    if (h, w) == (512, 512):
        return base_dir
    return os.path.join(base_dir, f"{h}x{w}")


def resize_tensor_hw(x, target_h, target_w, mode="bilinear"):
    """
    Resize a 4D tensor `(B, C, H, W)` to `(target_h, target_w)` without
    cropping.
    """
    if x.ndim != 4:
        raise ValueError(f"resize_tensor_hw expects a 4D tensor. Received shape={tuple(x.shape)}")

    # Keep interpolation handling explicit so the same helper works for both
    # images and masks.
    if mode in {"bilinear", "bicubic", "trilinear", "linear"}:
        return F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=False)
    return F.interpolate(x, size=(target_h, target_w), mode=mode)

def configure_cross_attention_runtime_resolution(model_or_unet, image_h, image_w, f):
    """
    Inject the base latent resolution so self-attention layers can recover
    rectangular `(h, w)` shapes from token counts.
    """
    latent_hw = (image_h // f, image_w // f)
    for m in model_or_unet.modules():
        if isinstance(m, CrossAttention):
            m.base_latent_hw = latent_hw


def validate_precomputed_z_enc_shape(name, z_enc, target_h, target_w, f):
    """
    Verify that a precomputed latent feature matches the active runtime
    resolution.
    """
    if z_enc is None:
        raise FileNotFoundError(f"Missing precomputed z_enc: {name}")

    expected_h = target_h // f
    expected_w = target_w // f
    actual_h, actual_w = z_enc.shape[-2], z_enc.shape[-1]
    if (actual_h, actual_w) != (expected_h, expected_w):
        raise ValueError(
            f"Precomputed z_enc resolution mismatch: {name} "
            f"(expected latent=({expected_h}, {expected_w}), actual=({actual_h}, {actual_w}))"
        )


def resolve_meta_path(data_root, meta_file):
    """
    Resolve the meta file path for `run_ori.py`.

    - Absolute paths are used as-is.
    - Relative paths are resolved against `data_root`.
    """
    if os.path.isabs(meta_file):
        return meta_file
    return os.path.join(data_root, meta_file)


def _resolve_content_token_path(data_root, token):
    """
    Resolve a content token from the canonical meta format.

    Supported forms:
    - `content_001.jpg` -> `{data_root}/cnt/content_001.jpg`
    - `cnt/content_001.jpg` -> `{data_root}/cnt/content_001.jpg`
    - absolute path -> unchanged
    """
    if os.path.isabs(token):
        return token
    if token.startswith("cnt/"):
        return os.path.join(data_root, token)
    return os.path.join(data_root, "cnt", token)


def _resolve_style_token_path(data_root, token):
    """
    Resolve a style token from the canonical meta format.

    Supported forms:
    - absolute path
    - `sty/...` relative to `data_root`
    - explicit subpaths such as `char/...` or `back/...`, resolved under
      `{data_root}/sty`

    Plain filenames are intentionally rejected so the official runtime contract
    stays explicit and order-independent.
    """
    if os.path.isabs(token):
        return token

    if token.startswith("sty/"):
        return os.path.join(data_root, token)

    if "/" in token:
        return os.path.join(data_root, "sty", token)

    raise ValueError(
        "Style tokens must be explicit paths such as 'sty/foo.jpg', "
        "'char/foo.jpg', or an absolute path. "
        f"Received '{token}'."
    )


def parse_multistyle_meta_samples(data_root, meta_file):
    """
    Parse the canonical multi-style meta file.

    Each non-empty, non-comment line must follow:
    `<content_token> <style_token_1> <style_token_2> ... <style_token_N>`
    """
    meta_path = resolve_meta_path(data_root, meta_file)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    samples = []
    with open(meta_path, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if (not line) or line.startswith("#"):
                continue

            tokens = line.split()

            cnt_token = tokens[0]
            style_tokens = tokens[1:]
            cnt_path = _resolve_content_token_path(data_root, cnt_token)
            style_paths = [
                _resolve_style_token_path(data_root, token)
                for token in style_tokens
            ]

            if not os.path.isfile(cnt_path):
                raise FileNotFoundError(f"Content image not found (line {line_no}): {cnt_path}")
            for style_path in style_paths:
                if not os.path.isfile(style_path):
                    raise FileNotFoundError(f"Style image not found (line {line_no}): {style_path}")

            samples.append({
                "cnt_path": cnt_path,
                "style_paths": style_paths,
            })

    # Preserve the file order exactly so the meta line order always defines the
    # style branch order.
    return samples


def prepare_style_mask_tensor_for_runtime(mask_npy_path, target_h, target_w, device):
    """
    Load a style mask file (`_mask{i}.npy`) and resize it to the runtime image
    resolution.
    """
    mask_np = np.load(mask_npy_path).astype(np.float32)
    if mask_np.ndim != 2:
        raise ValueError(f"Style masks must be 2D npy files: {mask_npy_path}, shape={mask_np.shape}")

    if mask_np.max() > 1.0:
        mask_np = mask_np / 255.0
    mask_np = np.clip(mask_np, 0.0, 1.0)

    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    mask = resize_tensor_hw(mask, target_h, target_w, mode="bilinear")
    return mask.clamp_(0.0, 1.0)


def _multimask_path_list_from_content(cnt_path, num_styles):
    """
    Build the expected `_mask{i}.npy` paths for a content image.
    """
    if num_styles < 1:
        raise ValueError(f"num_styles must be at least 1. Received {num_styles}.")

    base = os.path.splitext(cnt_path)[0]
    return [f"{base}_mask{style_idx}.npy" for style_idx in range(num_styles)]


def load_style_mask_stack_for_runtime(cnt_path, target_h, target_w, num_styles, device):
    """
    Load `_mask0.npy ... _mask{N-1}.npy` and stack them at runtime resolution.
    """
    mask_paths = _multimask_path_list_from_content(cnt_path, num_styles)
    missing_paths = [p for p in mask_paths if not os.path.isfile(p)]
    if missing_paths:
        raise FileNotFoundError(
            "Missing N-style mask files. "
            f"Expected files like {mask_paths[0]} ... {mask_paths[-1]}. "
            f"Missing files: {missing_paths}"
        )

    mask_tensors = []
    for mask_path in mask_paths:
        mask_tensors.append(
            prepare_style_mask_tensor_for_runtime(
                mask_path,
                target_h=target_h,
                target_w=target_w,
                device=device,
            )
        )
    return torch.cat(mask_tensors, dim=0)  # (N,1,H,W)


def build_runtime_style_mask_stack(cnt_path, target_h, target_w, num_styles, device, single_style_mode=None):
    """
    Build the runtime style mask stack for a sample.

    - `single_style_mode == "global"` uses an all-ones weight map and skips
      file loading.
    - `single_style_mode == "masked"` thresholds the single style mask to keep
      a hard content fallback outside mask coverage.
    """
    if single_style_mode == "global":
        return torch.ones((1, 1, target_h, target_w), device=device, dtype=torch.float32)

    style_mask_stack = load_style_mask_stack_for_runtime(
        cnt_path=cnt_path,
        target_h=target_h,
        target_w=target_w,
        num_styles=num_styles,
        device=device,
    )
    if single_style_mode == "masked" and num_styles == 1:
        style_mask_stack = (style_mask_stack >= 0.5).to(dtype=torch.float32)
    return style_mask_stack


def build_style_weight_maps_for_latent_from_masks(style_mask_stack_img, latent_h, latent_w):
    """
    Convert a runtime-resolution mask stack `(N, 1, H, W)` into latent-space
    style weights and coverage.

    - Overlapping regions are normalized across styles.
    - Empty regions keep `coverage == 0` so the caller can fall back to the
      content latent.
    """
    raw_weights = F.interpolate(
        style_mask_stack_img,
        size=(latent_h, latent_w),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    weight_sum = raw_weights.sum(dim=0, keepdim=True)  # (1,1,h,w)
    coverage = weight_sum.clamp(0.0, 1.0)
    norm_weights = raw_weights / weight_sum.clamp_min(1e-6)
    return norm_weights, coverage


def resolve_single_style_mode(opt, num_styles):
    """Resolve the optional single-style mode for the current sample."""
    if num_styles != 1:
        if opt.single_ver1 or opt.single_ver2:
            raise ValueError(
                "Single-style flags may only be used when the sample contains exactly one style image."
            )
        return None

    if opt.single_ver1:
        return "masked"
    if opt.single_ver2:
        return "global"
    return None


def build_adain_init_latent(cnt_z_enc, style_z_enc_list, style_mask_stack_img, device):
    """
    Build the initial latent by AdaIN-transforming the content latent toward
    each style latent and blending the result with normalized N-style masks.
    """
    cnt_z_enc_dev = cnt_z_enc.to(device)
    latent_h, latent_w = cnt_z_enc.shape[2], cnt_z_enc.shape[3]
    style_weights_latent, style_coverage_latent = build_style_weight_maps_for_latent_from_masks(
        style_mask_stack_img,
        latent_h,
        latent_w,
    )
    style_weights_latent = style_weights_latent.to(device=device, dtype=cnt_z_enc.dtype)
    style_coverage_latent = style_coverage_latent.to(device=device, dtype=cnt_z_enc.dtype)

    adain_components = [
        adain(cnt_z_enc_dev, style_z_enc.to(device))
        for style_z_enc in style_z_enc_list
    ]
    adain_stack = torch.stack(adain_components, dim=0)  # (N, B, C, h, w)
    styled_latent = (style_weights_latent.unsqueeze(1) * adain_stack).sum(dim=0)
    adain_z_enc = (
        styled_latent * style_coverage_latent +
        cnt_z_enc_dev * (1.0 - style_coverage_latent)
    ).clone().detach()

    del style_weights_latent, style_coverage_latent
    del adain_components, adain_stack, styled_latent, cnt_z_enc_dev
    return adain_z_enc

def get_cpu_mem():
    """Return resident CPU memory usage in MB."""
    return process.memory_info().rss / 1024 ** 2

def save_img_from_sample(model, samples_ddim, fname):
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(fname)

def load_minimal_feat(feat_path):
    """Load only the minimal feature subset needed at inference time."""
    with open(feat_path, 'rb') as h:
        feat_full = pickle.load(h)

    z_enc = None
    if feat_full[0] is not None and 'z_enc' in feat_full[0]:
        z_enc = feat_full[0]['z_enc'].clone().detach().cpu()

    feat_minimal = []
    for item in feat_full:
        if item is None:
            feat_minimal.append(None)
        else:
            minimal_dict = {}
            for key in item.keys():
                # Only keep the tensors required by the injection merge step.
                if any(key.endswith(suffix) for suffix in ['q', 'k', 'v', '_cnt_h']):
                    if torch.is_tensor(item[key]):
                        minimal_dict[key] = item[key].clone().detach().cpu()
                    else:
                        minimal_dict[key] = item[key]
            feat_minimal.append(minimal_dict)

    del feat_full
    gc.collect()

    return z_enc, feat_minimal

def clone_feat_value(value):
    """Clone tensors while leaving non-tensor metadata unchanged."""
    return value.clone().detach() if torch.is_tensor(value) else value


def build_injected_feature_maps(opt, cnt_feats, style_feats_list, single_style_mode=None):
    """
    Build the unified injected feature schema for any `num_styles >= 1`.

    Contract:
    - content q/k/v -> `..._cnt`
    - style j k/v   -> `..._sty{j}` for `j in [1, N]`
    - config always carries `num_styles` and `pi_style_total`
    """
    if len(style_feats_list) == 0:
        raise ValueError("style_feats_list must contain at least one style feature set.")

    num_styles = len(style_feats_list)
    merged_maps = []

    for i in range(len(cnt_feats)):
        feat_map = {
            'config': {
                'gamma': opt.gamma,
                'T': opt.T,
                'timestep': i,
                'num_styles': num_styles,
                'pi_style_total': opt.pi_style_total,
            }
        }
        if single_style_mode is not None:
            feat_map['config']['single_style_mode'] = single_style_mode

        cnt_feat = cnt_feats[i]
        style_feats_at_i = [style_feats[i] for style_feats in style_feats_list]

        if cnt_feat is None or any(sf is None for sf in style_feats_at_i):
            merged_maps.append(feat_map)
            continue

        for ori_key in cnt_feat.keys():
            if ori_key.endswith(('q', 'k', 'v')):
                feat_map[f"{ori_key}_cnt"] = clone_feat_value(cnt_feat[ori_key])

            if ori_key.endswith(('k', 'v')):
                for style_idx_one_based, style_feat in enumerate(style_feats_at_i, start=1):
                    feat_map[f"{ori_key}_sty{style_idx_one_based}"] = clone_feat_value(style_feat[ori_key])

        merged_maps.append(feat_map)

    return merged_maps

def adain(cnt_feat, sty_feat):
    with torch.no_grad():
        cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
        cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
        sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
        sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
        return ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model

def clear_all_caches(model):
    """Clear caches and temporary attributes from injected modules."""
    for m in model.modules():
        if isinstance(m, CrossAttention):
            if hasattr(m, "mask_cache"):
                m.mask_cache = {}
            if hasattr(m, 'cnt_name'):
                delattr(m, 'cnt_name')

        if m.__class__.__name__.endswith("ResBlock"):
            attrs_to_remove = []
            for attr_name in dir(m):
                if not attr_name.startswith('_') and not callable(getattr(m, attr_name)):
                    if any(keyword in attr_name for keyword in ['feat', 'cache', 'timestep', 'inject']):
                        attrs_to_remove.append(attr_name)
            
            for attr_name in attrs_to_remove:
                try:
                    delattr(m, attr_name)
                except:
                    pass

def move_feat_maps_to_device_inplace(feat_maps, device):
    """Move injected feature maps to the target device in place."""
    for i, f in enumerate(feat_maps):
        if isinstance(f, dict):
            keys_to_update = []
            for k, v in f.items():
                if torch.is_tensor(v) and v.device != device:
                    keys_to_update.append(k)
            
            for k in keys_to_update:
                v = f[k]
                f[k] = v.to(device, non_blocking=False)
                del v
                
        elif torch.is_tensor(f) and f.device != device:
            feat_maps[i] = f.to(device, non_blocking=False)
    
    gc.collect()
    return feat_maps

def restore_original_forwards(unet):
    """Restore the original `forward` methods after a stylization run."""
    for block_id in range(6, 12):
        if block_id >= len(unet.output_blocks):
            break
        for module in reversed(unet.output_blocks[block_id]):
            if module.__class__.__name__.endswith("ResBlock"):
                if hasattr(module, '_original_forward_backup'):
                    module._forward = module._original_forward_backup
                attrs_to_remove = ['block_id', 'ri_timestep']
                for attr in attrs_to_remove:
                    if hasattr(module, attr):
                        delattr(module, attr)
                break

def backup_original_forwards(unet):
    """Back up original `forward` methods before patching the decoder blocks."""
    for block_id in range(6, 12):
        if block_id >= len(unet.output_blocks):
            break
        for module in reversed(unet.output_blocks[block_id]):
            if module.__class__.__name__.endswith("ResBlock"):
                if not hasattr(module, '_original_forward_backup'):
                    module._original_forward_backup = module._forward
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddim_inv_steps', type=int, default=50)
    parser.add_argument('--save_feat_steps', type=int, default=50)
    parser.add_argument('--start_step', type=int, default=49)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--H', type=int, default=None)
    parser.add_argument('--W', type=int, default=None)
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument('--T', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument(
        '--pi_style_total',
        type=float,
        default=0.9,
        help='Total style probability mass. The content branch receives 1 - pi_style_total.',
    )
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11')
    parser.add_argument('--model_config', type=str, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument('--precomputed', type=str, default=DEFAULT_PRECOMPUTED_DIR)
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    parser.add_argument('--precision', type=str, default='autocast')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    parser.add_argument("--ratio", default=0.3, type=float)
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        '--meta_file',
        type=str,
        default=DEFAULT_META_FILE,
        help='Canonical meta file with one content token followed by N explicit style tokens per line.',
    )
    parser.add_argument(
        '--single_ver1',
        action='store_true',
        help='For N=1, mix style/content only inside mask0 and keep pure content outside it.',
    )
    parser.add_argument(
        '--single_ver2',
        action='store_true',
        help='For N=1, ignore mask files and mix style/content globally.',
    )
    opt = parser.parse_args()

    if opt.single_ver1 and opt.single_ver2:
        raise ValueError("--single_ver1 and --single_ver2 cannot be used at the same time.")

    seed_everything(opt.seed)
    opt.model_config = os.path.abspath(opt.model_config)
    opt.ckpt = os.path.abspath(opt.ckpt)
    opt.data_root = os.path.abspath(opt.data_root)
    opt.precomputed = os.path.abspath(opt.precomputed)
    opt.output_path = os.path.abspath(opt.output_path)
    os.makedirs(opt.output_path, exist_ok=True)

    model_config = OmegaConf.load(f"{opt.model_config}")
    opt.H, opt.W = resolve_runtime_resolution(opt, model_config)
    validate_runtime_resolution(opt.H, opt.W, opt.f)
    print(f"Runtime resolution: H={opt.H}, W={opt.W}")

    precomputed_dir = get_resolution_scoped_precomputed_dir(opt.precomputed, opt.H, opt.W)
    os.makedirs(precomputed_dir, exist_ok=True)

    model = load_model_from_config(model_config, f"{opt.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.save_feat_steps, ddim_eta=opt.ddim_eta, verbose=False)
    print("DDIM timesteps:", sampler.ddim_timesteps) 
    
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    samples_meta = parse_multistyle_meta_samples(opt.data_root, opt.meta_file)
    print(f"Loaded {len(samples_meta)} samples from meta: {resolve_meta_path(opt.data_root, opt.meta_file)}")

    unet_model = model.model.diffusion_model
    configure_cross_attention_runtime_resolution(unet_model, opt.H, opt.W, opt.f)

    backup_original_forwards(unet_model)
    if hasattr(model, "model_ema"):
        configure_cross_attention_runtime_resolution(model.model_ema.diffusion_model, opt.H, opt.W, opt.f)
        backup_original_forwards(model.model_ema.diffusion_model)
    
    begin = time.time()
    
    def residual_injection_callback(step_idx):
        t = sampler.ddim_timesteps[step_idx]
        for block_id in range(6, 12):
            if block_id >= len(unet_model.output_blocks):
                break
            for module in reversed(unet_model.output_blocks[block_id]):
                if module.__class__.__name__.endswith("ResBlock"):
                    module.ri_timestep = int(t)
                    break

        if hasattr(model, "model_ema"):
            ema_unet = model.model_ema.diffusion_model
            for block_id in range(6, 12):
                if block_id >= len(ema_unet.output_blocks):
                    break
                for module in reversed(ema_unet.output_blocks[block_id]):
                    if module.__class__.__name__.endswith("ResBlock"):
                        module.ri_timestep = int(t)
                        break

    for batch_idx, sample_meta in enumerate(samples_meta):
        print(f"Processing image {batch_idx+1}/{len(samples_meta)}")

        cnt_path = sample_meta["cnt_path"]
        style_paths = sample_meta["style_paths"]
        if len(style_paths) < 1:
            raise ValueError(f"Each sample must contain at least one style path: {sample_meta}")
        single_style_mode = resolve_single_style_mode(opt, len(style_paths))

        with torch.no_grad():
            for m in unet_model.modules():
                if isinstance(m, CrossAttention):
                    m.cnt_name = cnt_path

            cnt_base = os.path.splitext(os.path.basename(cnt_path))[0]
            style_bases = [os.path.splitext(os.path.basename(p))[0] for p in style_paths]

            print(f"Iteration {batch_idx}, CPU RAM: {get_cpu_mem():.2f} MB")

            style_z_enc_list = []
            style_feat_list = []
            for style_path in style_paths:
                style_base = os.path.splitext(os.path.basename(style_path))[0]
                style_feat_name = os.path.join(precomputed_dir, f"{style_base}_sty.pkl")
                if not os.path.isfile(style_feat_name):
                    raise FileNotFoundError(f"Style precomputed feature not found: {style_feat_name}")

                style_z_enc, style_feat = load_minimal_feat(style_feat_name)
                validate_precomputed_z_enc_shape(style_feat_name, style_z_enc, opt.H, opt.W, opt.f)
                style_z_enc_list.append(style_z_enc)
                style_feat_list.append(style_feat)

            cnt_feat_name = os.path.join(precomputed_dir, f"{cnt_base}_cnt.pkl")
            cnt_z_enc, cnt_feat = None, None
            if os.path.isfile(cnt_feat_name):
                cnt_z_enc, cnt_feat = load_minimal_feat(cnt_feat_name)
                validate_precomputed_z_enc_shape(cnt_feat_name, cnt_z_enc, opt.H, opt.W, opt.f)
            else:
                raise FileNotFoundError(f"Content precomputed feature not found: {cnt_feat_name}")
            
            print(f"After loading, CPU RAM: {get_cpu_mem():.2f} MB")

            schedule = make_content_injection_schedule(sampler.ddim_timesteps)
            cnt_feat_copy = copy.deepcopy(cnt_feat)
            patch_decoder_resblocks_h_and_cnt_hf(unet_model, schedule, cnt_feat_copy, ratio=opt.ratio)
            if hasattr(model, "model_ema"):
                patch_decoder_resblocks_h_and_cnt_hf(
                    model.model_ema.diffusion_model,
                    schedule,
                    cnt_feat_copy,
                    ratio=opt.ratio,
                )
            del cnt_feat_copy
            gc.collect()

            if opt.without_init_adain:
                adain_z_enc = cnt_z_enc.clone().detach()
            else:
                style_mask_stack_img = build_runtime_style_mask_stack(
                    cnt_path=cnt_path,
                    target_h=opt.H,
                    target_w=opt.W,
                    num_styles=len(style_z_enc_list),
                    device=device,
                    single_style_mode=single_style_mode,
                )
                adain_z_enc = build_adain_init_latent(
                    cnt_z_enc=cnt_z_enc,
                    style_z_enc_list=style_z_enc_list,
                    style_mask_stack_img=style_mask_stack_img,
                    device=device,
                )
                del style_mask_stack_img
                torch.cuda.empty_cache()

            del style_z_enc_list, cnt_z_enc
            gc.collect()

            if opt.without_attn_injection:
                merged_feat_maps = []
            else:
                merged_feat_maps = build_injected_feature_maps(
                    opt,
                    cnt_feat,
                    style_feat_list,
                    single_style_mode=single_style_mode,
                )
                merged_feat_maps = move_feat_maps_to_device_inplace(merged_feat_maps, device)

            del style_feat_list, cnt_feat
            gc.collect()

            print(f"Before inference, CPU RAM: {get_cpu_mem():.2f} MB")

            samples_ddim, _ = sampler.sample(
                S=opt.save_feat_steps,
                batch_size=1,
                shape=shape,
                verbose=False,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=adain_z_enc,
                injected_features=merged_feat_maps,
                start_step=opt.start_step,
                callback=residual_injection_callback,
            )

            result_name = f"{cnt_base}__{'__'.join(style_bases)}.png"
            save_img_from_sample(model, samples_ddim, os.path.join(opt.output_path, result_name))

        restore_original_forwards(unet_model)
        if hasattr(model, "model_ema"):
            restore_original_forwards(model.model_ema.diffusion_model)

        del samples_ddim, adain_z_enc, merged_feat_maps

        clear_all_caches(unet_model)
        if hasattr(model, "model_ema"):
            clear_all_caches(model.model_ema.diffusion_model)

        for _ in range(3):
            gc.collect()

        torch.cuda.empty_cache()

        if sys.platform == 'linux':
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except:
                pass

    print(f"Total time: {time.time() - begin:.2f} seconds")

if __name__ == "__main__":
    main()
