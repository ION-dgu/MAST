
from inspect import isfunction
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # print(f"context_dim is exists: {exists(context_dim)}")
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
      
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.q = None
        self.k = None
        self.v = None
        # Enable the unmodified path when the module is used to generate PKL features.
        self.gen_pkl = False

        self.cnt_name = None


        self.mask_cache = {}
        # The run script injects the base latent resolution so each attention
        # layer can recover its own rectangular spatial shape from token count.
        self.base_latent_hw = None

    def _infer_spatial_hw_from_tokens(self, token_count):
        """Recover `(h, w)` from a self-attention token count."""
        base_hw = getattr(self, "base_latent_hw", None)
        if base_hw is not None:
            base_h, base_w = int(base_hw[0]), int(base_hw[1])
            base_tokens = base_h * base_w

            if token_count <= 0:
                raise ValueError(f"Invalid token_count: {token_count}")

            if base_tokens % token_count == 0:
                area_ratio = base_tokens // token_count
                scale = math.isqrt(area_ratio)
                if scale * scale == area_ratio and scale > 0:
                    if base_h % scale == 0 and base_w % scale == 0:
                        h = base_h // scale
                        w = base_w // scale
                        if h * w == token_count:
                            return h, w

        side = int(math.isqrt(token_count))
        if side * side == token_count:
            return side, side

        raise ValueError(
            f"Failed to recover spatial shape from token_count={token_count}. "
            f"base_latent_hw={base_hw}"
        )

    def _load_and_preprocess_style_mask_file(self, mask_path, target_h, target_w, device):
        """Load a single `_mask{i}.npy` file and resize it to attention resolution."""
        mask = torch.tensor(np.load(mask_path), dtype=torch.float32, device=device)
        if mask.ndim != 2:
            raise ValueError(f"Style mask must be a 2D npy file: {mask_path}, shape={tuple(mask.shape)}")

        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = mask.clamp(0.0, 1.0)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return mask.squeeze(0).permute(1, 2, 0).contiguous().clamp_(0.0, 1.0)

    def _load_style_weight_maps_from_mask_files(self, mask_prefix, target_h, target_w, num_styles, device):
        """Load `_mask0.npy ... _mask{N-1}.npy` as attention-resolution style weights."""
        cache_key = (
            f"multi_mask_weights::{mask_prefix}::{target_h}::{target_w}::{num_styles}::"
            f"{getattr(self, 'base_latent_hw', None)}"
        )
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        mask_paths = [f"{mask_prefix}{style_idx}.npy" for style_idx in range(num_styles)]
        missing_paths = [p for p in mask_paths if not os.path.isfile(p)]
        if missing_paths:
            raise FileNotFoundError(
                "Missing N-style mask files. "
                f"Expected files like {mask_paths[0]} ... {mask_paths[-1]}. "
                f"Missing files: {missing_paths}"
            )

        weight_maps = torch.stack(
            [
                self._load_and_preprocess_style_mask_file(
                    mask_path=mask_path,
                    target_h=target_h,
                    target_w=target_w,
                    device=device,
                )
                for mask_path in mask_paths
            ],
            dim=0,
        ).contiguous()
        self.mask_cache[cache_key] = weight_maps
        return weight_maps

    def get_batch_sim(self, q, k, num_heads):
        """Compute scaled attention logits with shape `(heads, Nq, Nk)`."""
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        return torch.einsum("h i d, h j d -> h i j", q, k) * self.scale


    def apply_pi_mass_nway(self, style_sims, cc_sim, style_weight_maps, pi_style_total=0.9, eps=1e-6):
        """
        Implement LAMA for the N-style case by adding one shared bias per style
        logit group.

        Paper notation:
        - `style_sims[i]` corresponds to the pre-bias style logits
          `ell_cs^(i)(q)` for style partition `i`.
        - `cc_sim` corresponds to the content logits `ell_c(q)`.
        - `style_weight_maps[i]` corresponds to the resized continuous mask
          `M^(i)(q)` at each query location `q`.
        - `pi_style_total` is the global style mass budget `pi*`.

        The goal is the same as in the main paper Eq. (6)-(7) and Appendix A.1:
        after concatenating all style partitions and the content partition, the
        softmax should allocate a prescribed amount of attention probability mass
        to each style region while leaving the remaining mass to the content
        partition. The additive bias is constant within each partition, so the
        relative ordering inside that partition is preserved.
        """
        if len(style_sims) == 0:
            return []

        H, Nq, Nk = cc_sim.shape
        num_styles = len(style_sims)
        h = int(style_weight_maps.shape[1])
        w = int(style_weight_maps.shape[2])
        # Each query token is interpreted as one spatial location q. We reshape
        # `(heads, Nq, Nk)` into `(heads, h, w, Nk)` so the per-pixel masks
        # `M^(i)(q)` can be applied at the same spatial locations as in the paper.
        assert h * w == Nq and Nk == Nq, (
            f"style_weight_maps spatial shape does not match the attention tokens: "
            f"weights=({num_styles},{h},{w}), Nq={Nq}, Nk={Nk}"
        )

        # `weights_raw` stores the resized soft masks. In the paper,
        # `pi_cs^(i)(q) = pi* * M^(i)(q)`. In code, interpolation can introduce
        # soft values and local overlap between masks, so we first recover the
        # raw mask stack and then normalize it into per-style proportions.
        weights_raw = style_weight_maps.to(device=cc_sim.device, dtype=cc_sim.dtype).clamp_min(0.0)
        weight_sum = weights_raw.sum(dim=0, keepdim=True)
        # `coverage` measures how much of the current query location is claimed
        # by any style mask. Uncovered regions keep content mass close to 1.
        # When masks overlap after resizing, clamping keeps the total style mass
        # from exceeding the global budget `pi*`.
        coverage = weight_sum.clamp(0.0, 1.0)
        # `weights` are the normalized per-style proportions at location q.
        # If the masks are disjoint and sum to at most 1, this reduces exactly to
        # the paper's `M^(i)(q)`. When masks overlap, the proportions are
        # renormalized but the total style budget is still conserved.
        weights = weights_raw / weight_sum.clamp_min(eps)

        pi_style_total_t = torch.tensor(pi_style_total, device=cc_sim.device, dtype=cc_sim.dtype).clamp(eps, 1 - eps)
        # Effective style mass at each location:
        #   pi_style_total_eff(q) = pi* * coverage(q)
        # and the remaining content mass:
        #   pi_c(q) = 1 - pi_style_total_eff(q)
        #
        # Therefore:
        # - in fully uncovered regions, style mass becomes 0 and content keeps
        #   the full probability mass;
        # - in fully covered regions, the total style mass becomes `pi*`;
        # - in partial/soft regions, the style mass scales continuously.
        pi_style_total_eff = (pi_style_total_t * coverage).clamp(0.0, 1.0 - eps)
        pi_c = (1.0 - pi_style_total_eff).clamp(eps, 1 - eps)

        # `logZc` is the log partition function of the content group, i.e.
        # `log Z_c(q)` in Appendix A.1.
        sc = cc_sim.reshape(H, h, w, Nq)
        logZc = torch.logsumexp(sc, dim=3, keepdim=True)

        adjusted = []
        for style_idx, sim in enumerate(style_sims):
            sj = sim.reshape(H, h, w, Nq)
            # `logZj` is `log Z_cs^(i)(q)` for style partition `i`.
            logZj = torch.logsumexp(sj, dim=3, keepdim=True)
            # Target mass for style partition `i`:
            #   pi_i(q) = pi_style_total_eff(q) * normalized_mask_i(q)
            #
            # If masks are non-overlapping, this becomes exactly
            # `pi* * M^(i)(q)` from the paper.
            pi_j = (pi_style_total_eff * weights[style_idx:style_idx + 1]).clamp_min(eps)
            # Closed-form bias for the whole style partition:
            #   b^(i)(q) = log(pi_i(q) / pi_c(q)) + log Z_c(q) - log Z_cs^(i)(q)
            #
            # This is the N-style counterpart of the single-style derivation in
            # Appendix A.1.3. Adding the same bias to every token inside the
            # partition changes only the total mass of that partition, not the
            # intra-partition ranking.
            bj = logZc + (torch.log(pi_j) - torch.log(pi_c)) - logZj
            adjusted.append((sj + bj).reshape(H, Nq, Nq))

        return adjusted

    def _apply_groupwise_temperature(self, cat_sim, cc_sim):
        """
        Apply Sharpness-aware Temperature Scaling (STS).

        STS is proposed to address the degradation of attention sharpness caused by:
        1) Low correlation between content-style queries and style keys → flattened logits
        2) Concatenation of multiple partitions → increased softmax denominator → reduced contrast

        ------------------------------------------------------------
        Core idea:
        Instead of using a fixed temperature (as in StyleID),
        we adaptively compute temperature τ per head based on sharpness.

        Sharpness is measured using:
            log p_max = max(logits) - logsumexp(logits)

        We define the sharpness gap:
            Δ = log p_max(content logits) - log p_max(concatenated logits)

        Interpretation:
        - Δ ↑ → concatenation made distribution flatter → need larger τ
        - Δ ↓ → already sharp → small τ is sufficient

        ------------------------------------------------------------
        Implementation:
        - Compute Δ per head
        - Map Δ → τ using a polynomial fit:
            τ = aΔ² + bΔ + c
        - Apply temperature scaling:
            τ * (logits - mean) + mean

        NOTE:
        - Mean is preserved to maintain group-wise normalization stability
        - τ is clipped to prevent over-sharpening

        This restores attention sharpness and improves stylization fidelity.
        """

        H, _, _ = cat_sim.shape

        def log_pmax(logits, dim=-1):
            max_logit, _ = logits.max(dim=dim, keepdim=True)
            lse = torch.logsumexp(logits, dim=dim, keepdim=True)
            return (max_logit - lse).squeeze(dim)

        # Sharpness of content-only attention
        logp_cc = log_pmax(cc_sim)

        # Sharpness after concatenating style + content
        logp_cat = log_pmax(cat_sim)

        # Δ: sharpness gap (per head)
        delta_head = (logp_cc - logp_cat).mean(dim=1)

        # Polynomial mapping Δ → τ (fitted offline)
        a1 = 0.08395199
        b2 = 0.43704639
        c3 = 1.00998177

        tau = a1 * delta_head**2 + b2 * delta_head + c3

        # Prevent extreme scaling
        tau = torch.clamp(tau, min=1.0, max=5.0).view(H, 1, 1)

        # Preserve mean while scaling variance (important!)
        mean = cat_sim.mean(dim=-1, keepdim=True)

        return tau * (cat_sim - mean) + mean

    def _forward_standard_attention(self, q, k, v, h, mask=None, attn_matrix_scale=1.0, apply_attn_scale=False):
        """Run the standard attention path used for cross-attention and fallbacks."""
        sim = einsum('b i d, b j d -> b i j', q, k)
        if apply_attn_scale:
            sim *= attn_matrix_scale
        sim *= self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return rearrange(out, '(b h) n d -> b n (h d)', h=h)

    def _has_complete_injection_payload(
        self,
        cnt_q_injected,
        cnt_k_injected,
        cnt_v_injected,
        style_k_injected_list,
        style_v_injected_list,
        num_styles,
    ):
        """Validate that the unified injected payload is complete for N-style attention."""
        if num_styles < 1:
            return False
        if cnt_q_injected is None or cnt_k_injected is None or cnt_v_injected is None:
            return False
        if style_k_injected_list is None or style_v_injected_list is None:
            return False
        if len(style_k_injected_list) < num_styles or len(style_v_injected_list) < num_styles:
            return False
        if any(style_k_injected_list[idx] is None or style_v_injected_list[idx] is None for idx in range(num_styles)):
            return False
        return True

    def forward(
        self,
        x,
        context=None,
        mask=None,
        cnt_q_injected=None,
        cnt_k_injected=None,
        cnt_v_injected=None,
        style_k_injected_list=None,
        style_v_injected_list=None,
        injection_config=None,
    ):
        h = self.heads
        b = x.shape[0]
        is_cross = context is not None

        attn_matrix_scale = 1.0
        q_mix = 0.0
        pi_style_total = 0.9
        num_styles = 0
        single_style_mode = None
        if injection_config is not None:
            attn_matrix_scale = injection_config.get('T', 1.0)
            q_mix = injection_config.get('gamma', 0.0)
            pi_style_total = injection_config.get('pi_style_total', 0.9)
            num_styles = int(injection_config.get('num_styles', 0))
            single_style_mode = injection_config.get('single_style_mode', None)

        context = default(context, x)
        q_live = rearrange(self.to_q(x), 'b n (h d) -> (b h) n d', h=h)
        k_live = rearrange(self.to_k(context), 'b m (h d) -> (b h) m d', h=h)
        v_live = rearrange(self.to_v(context), 'b m (h d) -> (b h) m d', h=h)

        self.q = q_live
        self.k = k_live
        self.v = v_live

        if not self.gen_pkl and not is_cross and self.cnt_name is not None:
            if self._has_complete_injection_payload(
                cnt_q_injected=cnt_q_injected,
                cnt_k_injected=cnt_k_injected,
                cnt_v_injected=cnt_v_injected,
                style_k_injected_list=style_k_injected_list,
                style_v_injected_list=style_v_injected_list,
                num_styles=num_styles,
            ):
                q_content = torch.cat([cnt_q_injected] * b, dim=0)
                q_mixed = q_content * q_mix + q_live * (1.0 - q_mix)
                k_content = torch.cat([cnt_k_injected] * b, dim=0)
                v_content = torch.cat([cnt_v_injected] * b, dim=0)
                style_k_branches = [torch.cat([style_k_injected_list[idx]] * b, dim=0) for idx in range(num_styles)]
                style_v_branches = [torch.cat([style_v_injected_list[idx]] * b, dim=0) for idx in range(num_styles)]

                cc_sim = self.get_batch_sim(q=q_content, k=k_content, num_heads=h)
                style_sims = [
                    self.get_batch_sim(
                        q=q_mixed,
                        k=style_k_branch,
                        num_heads=h,
                    )
                    for style_k_branch in style_k_branches
                ]

                layer_h, layer_w = self._infer_spatial_hw_from_tokens(q_mixed.shape[1])
                if single_style_mode == "global" and num_styles == 1:
                    style_weight_maps = torch.ones(
                        (1, layer_h, layer_w, 1),
                        device=q_mixed.device,
                        dtype=cc_sim.dtype,
                    )
                else:
                    mask_prefix = os.path.splitext(self.cnt_name)[0] + "_mask"
                    style_weight_maps = self._load_style_weight_maps_from_mask_files(
                        mask_prefix=mask_prefix,
                        target_h=layer_h,
                        target_w=layer_w,
                        num_styles=num_styles,
                        device=q_mixed.device,
                    )
                    if single_style_mode == "masked" and num_styles == 1:
                        style_weight_maps = (style_weight_maps >= 0.5).to(dtype=cc_sim.dtype)

                # LAMA is applied before the final softmax over the concatenated
                # `[style_1, ..., style_N, content]` token groups. The returned
                # style logits already include the closed-form per-group biases.
                style_sims = self.apply_pi_mass_nway(
                    style_sims=style_sims,
                    cc_sim=cc_sim,
                    style_weight_maps=style_weight_maps,
                    pi_style_total=pi_style_total,
                )
                # Concatenate style and content logits:
                #   ℓ_concat = [ℓ_cs^(1), ..., ℓ_cs^(N), ℓ_c]
                #
                # LAMA → controls attention mass (global allocation)
                # STS  → restores sharpness after concatenation
                cat_sim = self._apply_groupwise_temperature(torch.cat(style_sims + [cc_sim], dim=2), cc_sim)
                cat_v = torch.cat(style_v_branches + [v_content], dim=1)
                cat_attn = cat_sim.softmax(dim=-1)
                cat_out = einsum('b i j, b j d -> b i d', cat_attn, cat_v)
                out = rearrange(cat_out, 'h (b n) d -> b n (h d)', h=h, b=b)
                return self.to_out(out)

        ## when not injection layer.
        out = self._forward_standard_attention(
            q=q_live,
            k=k_live,
            v=v_live,
            h=h,
            mask=mask if self.gen_pkl else None,
            attn_matrix_scale=attn_matrix_scale,
            apply_attn_scale=(not is_cross and injection_config is not None and cnt_q_injected is not None),
        )
        
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
    def forward(self,
                x,
                context=None,
                self_attn_cnt_q_injected=None,
                self_attn_cnt_k_injected=None,
                self_attn_cnt_v_injected=None,
                self_attn_style_k_injected_list=None,
                self_attn_style_v_injected_list=None,
                injection_config=None,
                ):
        return checkpoint(self._forward, (x,
                                          context,
                                          self_attn_cnt_q_injected,
                                          self_attn_cnt_k_injected,
                                          self_attn_cnt_v_injected,
                                          self_attn_style_k_injected_list,
                                          self_attn_style_v_injected_list,
                                          injection_config,), self.parameters(), self.checkpoint)

    def _forward(self,
                 x,
                 context=None,
                 self_attn_cnt_q_injected=None,
                 self_attn_cnt_k_injected=None,
                 self_attn_cnt_v_injected=None,
                 self_attn_style_k_injected_list=None,
                 self_attn_style_v_injected_list=None,
                 injection_config=None):
        x_ = self.attn1(self.norm1(x),
                       cnt_q_injected=self_attn_cnt_q_injected,
                       cnt_k_injected=self_attn_cnt_k_injected,
                       cnt_v_injected=self_attn_cnt_v_injected,
                       style_k_injected_list=self_attn_style_k_injected_list,
                       style_v_injected_list=self_attn_style_v_injected_list,
                       injection_config=injection_config,)
        x = x_ + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self,
                x,
                context=None,
                self_attn_cnt_q_injected=None,
                self_attn_cnt_k_injected=None,
                self_attn_cnt_v_injected=None,
                self_attn_style_k_injected_list=None,
                self_attn_style_v_injected_list=None,
                injection_config=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x,
                      context=context,
                      self_attn_cnt_q_injected=self_attn_cnt_q_injected,
                      self_attn_cnt_k_injected=self_attn_cnt_k_injected,
                      self_attn_cnt_v_injected=self_attn_cnt_v_injected,
                      self_attn_style_k_injected_list=self_attn_style_k_injected_list,
                      self_attn_style_v_injected_list=self_attn_style_v_injected_list,
                      injection_config=injection_config)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
