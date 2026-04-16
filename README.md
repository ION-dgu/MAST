# MAST: Multi-Style Arbitrary Style Transfer

Official implementation of **MAST**, a multi-style arbitrary style transfer framework for **single-style**, **few-style**, and **N-style** stylization.

This pipeline unifies fixed 2-style setting into a general **1, 2, ..., N** style formulation.

---

## Overview

MAST takes:

- one **content image**
- one or more **style images**
- optional **region masks**
- precomputed feature **`.pkl` files**

It supports the following modes:

- **`single_ver1`**: single-style masked transfer using `mask0`
- **`single_ver2`**: single-style global transfer over the full image
- **multi-style transfer**: `2-style`, `3-style`, ..., `N-style`

For multi-style transfer, the order of style images must match the order of masks:

- `style_001 -> mask0`
- `style_002 -> mask1`
- `style_003 -> mask2`
- ...

Special cases:

- `single_ver1` uses only `mask0`
- `single_ver2` ignores masks entirely

---

## Data Preparation

A typical data layout is:

```text
data/
├── cnt/
│   ├── content_020.jpg
│   ├── content_020_mask0.png
│   ├── content_020_mask1.png
│   └── content_020_mask2.png
├── sty/
│   ├── style_001.jpg
│   ├── style_002.jpg
│   └── style_003.jpg
└── meta/
    ├── single_ver1.txt
    ├── single_ver2.txt
    ├── style2.txt
    └── style3.txt
```

### 1. Convert masks to `.npy`

```bash
python make_mask_npy.py \
  --mask_dir path/to/masks \
  --save_dir data/cnt
```

This converts `mask*.png` into `mask*.npy`.

### 2. Generate precomputed features

`generate_pkl_only.py` and `run_ori.py` must use the **same resolution**.

In this README, we use **512 × 512**:

```bash
python generate_pkl_only.py \
  --H 512 \
  --W 512 \
  --data_root data \
  --precomputed precomputed_feats
```

This generates:

- `content_xxx_cnt.pkl`
- `style_xxx_sty.pkl`

under `precomputed_feats/`.

---

## Meta File Format

Each line in the meta file should follow:

```text
<content_token> <style_token_1> <style_token_2> ... <style_token_N>
```

Examples:

### Single style

```text
cnt/content_020.jpg sty/style_001.jpg
```

### 2-style

```text
cnt/content_020.jpg sty/style_001.jpg sty/style_002.jpg
```

### 3-style

```text
cnt/content_020.jpg sty/style_001.jpg sty/style_002.jpg sty/style_003.jpg
```

Supported content tokens:

- `content_001.jpg`
- `cnt/content_001.jpg`
- absolute path

Supported style tokens:

- `sty/style_001.jpg`
- `char/style_001.jpg`
- `back/style_002.jpg`
- absolute path

---

## Inference

Common hyperparameters:

```bash
--gamma 0.2 --T 2 --ratio 0.3
```

### Single-style masked (`single_ver1`)

Uses one style image and `mask0` only.

```bash
python run_ori.py \
  --H 512 \
  --W 512 \
  --data_root data \
  --precomputed precomputed_feats \
  --meta_file meta/single_ver1.txt \
  --output_path output/single_ver1 \
  --gamma 0.2 \
  --T 2 \
  --ratio 0.3 \
  --single_ver1
```

### Single-style global (`single_ver2`)

Uses one style image and ignores masks.

```bash
python run_ori.py \
  --H 512 \
  --W 512 \
  --data_root data \
  --precomputed precomputed_feats \
  --meta_file meta/single_ver2.txt \
  --output_path output/single_ver2 \
  --gamma 0.2 \
  --T 2 \
  --ratio 0.3 \
  --single_ver2
```

### 2-style

Uses `style_001`, `style_002` with `mask0`, `mask1`.

```bash
python run_ori.py \
  --H 512 \
  --W 512 \
  --data_root data \
  --precomputed precomputed_feats \
  --meta_file meta/style2.txt \
  --output_path output/style2 \
  --gamma 0.2 \
  --T 2 \
  --ratio 0.3
```

### 3-style

Uses `style_001`, `style_002`, `style_003` with `mask0`, `mask1`, `mask2`.

```bash
python run_ori.py \
  --H 512 \
  --W 512 \
  --data_root data \
  --precomputed precomputed_feats \
  --meta_file meta/style3.txt \
  --output_path output/style3 \
  --gamma 0.2 \
  --T 2 \
  --ratio 0.3
```

The same rule naturally extends to **N-style** transfer.

---

## Output

The output filename is automatically saved as:

```text
{content_base}__{style1_base}__{style2_base}__...png
```

Examples:

```text
content_020__style_001.png
content_020__style_001__style_002.png
content_020__style_001__style_002__style_003.png
```

---

## Notes

- `generate_pkl_only.py` and `run_ori.py` must use the same `--H` and `--W`
- style order must match mask order
- `single_ver1` uses `mask0`
- `single_ver2` ignores masks
- the framework extends directly from 1-style to N-style

---

## Citation

If you use this code, please cite the corresponding paper.

```bibtex
@article{mast,
  title   = {MAST: Multi-Style Arbitrary Style Transfer},
  author  = {Author Names},
  journal = {Journal / Conference},
  year    = {202X}
}
```
