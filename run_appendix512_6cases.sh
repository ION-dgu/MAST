#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_DIR/appendix_512}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-$DATA_DIR/precomputed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_DIR/runcheck_outputs/deadcode_recheck}"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMPAT_DIR="${COMPAT_DIR:-/tmp/acmmm_compat}"

CONTENT_TOKEN="content_020.jpg"
STYLE_TOKENS=(
  "sty/style_001.jpg"
  "sty/style_002.jpg"
  "sty/style_003.jpg"
  "sty/style_004.jpg"
  "sty/style_005.jpg"
)

COMMON_ARGS=(
  --data_root "$DATA_DIR"
  --precomputed "$PRECOMPUTED_DIR"
  --gamma 0.2
  --T 2
  --ratio 0.3
)

ensure_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

write_compat_sitecustomize() {
  mkdir -p "$COMPAT_DIR"
  cat > "$COMPAT_DIR/sitecustomize.py" <<'PY'
import sys
import types

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except Exception:
    rank_zero_only = None

module = types.ModuleType("pytorch_lightning.utilities.distributed")
if rank_zero_only is not None:
    module.rank_zero_only = rank_zero_only
sys.modules.setdefault("pytorch_lightning.utilities.distributed", module)
PY
  export PYTHONPATH="$COMPAT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
}

prepare_meta_files() {
  META_DIR="$OUTPUT_ROOT/meta"
  mkdir -p "$META_DIR"

  cat > "$META_DIR/single_ver1.txt" <<EOF
$CONTENT_TOKEN ${STYLE_TOKENS[0]}
EOF

  cat > "$META_DIR/single_ver2.txt" <<EOF
$CONTENT_TOKEN ${STYLE_TOKENS[0]}
EOF

  cat > "$META_DIR/style2.txt" <<EOF
$CONTENT_TOKEN ${STYLE_TOKENS[0]} ${STYLE_TOKENS[1]}
EOF

  cat > "$META_DIR/style3.txt" <<EOF
$CONTENT_TOKEN ${STYLE_TOKENS[0]} ${STYLE_TOKENS[1]} ${STYLE_TOKENS[2]}
EOF

  cat > "$META_DIR/style4.txt" <<EOF
$CONTENT_TOKEN ${STYLE_TOKENS[0]} ${STYLE_TOKENS[1]} ${STYLE_TOKENS[2]} ${STYLE_TOKENS[3]}
EOF

  cat > "$META_DIR/style5.txt" <<EOF
$CONTENT_TOKEN ${STYLE_TOKENS[0]} ${STYLE_TOKENS[1]} ${STYLE_TOKENS[2]} ${STYLE_TOKENS[3]} ${STYLE_TOKENS[4]}
EOF
}

check_inputs() {
  ensure_file "$DATA_DIR/cnt/content_020.jpg"
  ensure_file "$DATA_DIR/cnt/content_020_mask0.npy"
  ensure_file "$DATA_DIR/cnt/content_020_mask1.npy"
  ensure_file "$DATA_DIR/cnt/content_020_mask2.npy"
  ensure_file "$DATA_DIR/cnt/content_020_mask3.npy"
  ensure_file "$DATA_DIR/cnt/content_020_mask4.npy"

  ensure_file "$DATA_DIR/sty/style_001.jpg"
  ensure_file "$DATA_DIR/sty/style_002.jpg"
  ensure_file "$DATA_DIR/sty/style_003.jpg"
  ensure_file "$DATA_DIR/sty/style_004.jpg"
  ensure_file "$DATA_DIR/sty/style_005.jpg"

  ensure_file "$PRECOMPUTED_DIR/content_020_cnt.pkl"
  ensure_file "$PRECOMPUTED_DIR/style_001_sty.pkl"
  ensure_file "$PRECOMPUTED_DIR/style_002_sty.pkl"
  ensure_file "$PRECOMPUTED_DIR/style_003_sty.pkl"
  ensure_file "$PRECOMPUTED_DIR/style_004_sty.pkl"
  ensure_file "$PRECOMPUTED_DIR/style_005_sty.pkl"
}

run_case() {
  local label="$1"
  local meta_file="$2"
  shift 2

  local out_dir="$OUTPUT_ROOT/$label"
  local log_dir="$OUTPUT_ROOT/logs"
  local log_file="$log_dir/${label}.log"

  mkdir -p "$out_dir" "$log_dir"

  echo
  echo "============================================================"
  echo "Running case: $label"
  echo "Meta file: $meta_file"
  echo "Output dir: $out_dir"
  echo "Log file: $log_file"
  echo "============================================================"

  "$PYTHON_BIN" "$REPO_DIR/run_ori.py" \
    "${COMMON_ARGS[@]}" \
    --meta_file "$meta_file" \
    --output_path "$out_dir" \
    "$@" 2>&1 | tee "$log_file"
}

main() {
  mkdir -p "$OUTPUT_ROOT"
  write_compat_sitecustomize
  check_inputs
  prepare_meta_files

  run_case "single_ver1" "$META_DIR/single_ver1.txt" --single_ver1
  run_case "single_ver2" "$META_DIR/single_ver2.txt" --single_ver2
  run_case "style2" "$META_DIR/style2.txt"
  run_case "style3" "$META_DIR/style3.txt"
  run_case "style4" "$META_DIR/style4.txt"
  run_case "style5" "$META_DIR/style5.txt"

  echo
  echo "All six appendix_512 inference cases completed."
  echo "Results: $OUTPUT_ROOT"
}

main "$@"
