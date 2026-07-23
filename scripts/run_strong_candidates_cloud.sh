#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential cloud launcher for the three pre-registered strong candidates.
# This script reuses scripts/train.py for every run and never opens locked test
# data. Override behavior with environment variables documented below.

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-2026 2027 2028}"
CANDIDATES="${CANDIDATES:-s1 s2 s3}"
SYNTHETIC_SAMPLES="${SYNTHETIC_SAMPLES:-170}"
SYNTHETIC_DATA_DIR="${SYNTHETIC_DATA_DIR:-$PROJECT_ROOT/synthetic_data/WADI-CLEAN_X_train_full_l64_seed2026}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/cloud_logs/strong_candidates_$RUN_TAG}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_ROOT"

required_package_files=(
  synthetic_windows.npy
  point_masks.npy
  channel_masks.npy
  endpoint_labels.npy
  source_endpoints.npy
  operator_codes.npy
  channel_metadata.json
  dataset_summary.json
)

if [[ ! -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi

for file_name in "${required_package_files[@]}"; do
  if [[ ! -f "$SYNTHETIC_DATA_DIR/$file_name" ]]; then
    echo "Synthetic package is incomplete: $SYNTHETIC_DATA_DIR/$file_name" >&2
    exit 1
  fi
done

read -r -a seed_list <<< "$SEEDS"
read -r -a candidate_list <<< "$CANDIDATES"
if [[ ${#seed_list[@]} -eq 0 || ${#candidate_list[@]} -eq 0 ]]; then
  echo "SEEDS and CANDIDATES must each contain at least one value." >&2
  exit 1
fi

candidate_config() {
  case "$1" in
    s1) echo "configs/wadi_s1_numeric_synth_verbalizer.json" ;;
    s2) echo "configs/wadi_s2_dtt_replacement_synth_verbalizer.json" ;;
    s3) echo "configs/wadi_s3_dtt_additive_synth_verbalizer.json" ;;
    *)
      echo "Unknown candidate '$1'; expected s1, s2, or s3." >&2
      return 1
      ;;
  esac
}

candidate_slug() {
  case "$1" in
    s1) echo "numeric" ;;
    s2) echo "dtt_replace" ;;
    s3) echo "dtt_add" ;;
    *) return 1 ;;
  esac
}

heartbeat_pid=""
cleanup_heartbeat() {
  if [[ -n "$heartbeat_pid" ]] && kill -0 "$heartbeat_pid" 2>/dev/null; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  heartbeat_pid=""
}
trap cleanup_heartbeat EXIT INT TERM

run_logged() {
  local log_file="$1"
  shift
  local -a command=("$@")

  {
    printf '[%s] command:' "$(date --iso-8601=seconds)"
    printf ' %q' "${command[@]}"
    printf '\n'
  } | tee -a "$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  (
    while sleep "$HEARTBEAT_SECONDS"; do
      printf '[%s] heartbeat: training is still running\n' "$(date --iso-8601=seconds)" |
        tee -a "$log_file"
    done
  ) &
  heartbeat_pid=$!

  set +e
  PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee -a "$log_file"
  local command_status=${PIPESTATUS[0]}
  set -e

  cleanup_heartbeat
  return "$command_status"
}

echo "Project root: $PROJECT_ROOT"
echo "Seeds: $SEEDS"
echo "Candidates: $CANDIDATES"
echo "Synthetic package: $SYNTHETIC_DATA_DIR"
echo "Synthetic windows per epoch: $SYNTHETIC_SAMPLES"
echo "Logs: $LOG_ROOT"
echo "Dry run: $DRY_RUN"

# Seed is the outer loop so every completed group contains paired S1/S2/S3
# results with identical seed-dependent endpoint and synthetic schedules.
for seed in "${seed_list[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 1
  fi
  for candidate in "${candidate_list[@]}"; do
    config="$(candidate_config "$candidate")"
    slug="$(candidate_slug "$candidate")"
    if [[ ! -f "$PROJECT_ROOT/$config" ]]; then
      echo "Candidate config is missing: $PROJECT_ROOT/$config" >&2
      exit 1
    fi

    run_name="${candidate}_${slug}_l64_l12_direct_vonly_synrot${SYNTHETIC_SAMPLES}_seed${seed}_${RUN_TAG}"
    output_dir="$PROJECT_ROOT/outputs/$run_name"
    log_file="$LOG_ROOT/$run_name.log"

    if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to reuse non-empty output directory: $output_dir" >&2
      echo "Choose a new RUN_TAG or remove only the explicitly reviewed failed run." >&2
      exit 1
    fi

    command=(
      "$PYTHON_BIN" -u scripts/train.py
      --config "$config"
      --run-name "$run_name"
      --seed "$seed"
      --use-synthetic-anomalies
      --synthetic-data-dir "$SYNTHETIC_DATA_DIR"
      --synthetic-samples "$SYNTHETIC_SAMPLES"
      --synthetic-sampling epoch_stratified
      --full-run-authorized
      --device "$DEVICE"
    )

    echo
    echo "Starting candidate=$candidate seed=$seed run=$run_name"
    if ! run_logged "$log_file" "${command[@]}"; then
      echo "Run failed: $run_name" >&2
      echo "Inspect log: $log_file" >&2
      exit 1
    fi
    echo "Completed: $run_name"
  done
done

echo
echo "All requested candidate runs completed successfully."
echo "Logs: $LOG_ROOT"
