#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential cloud launcher for the classifier-head development round.
#
# Default candidates for the semantic counterfactual round:
#   e4: additive compact DTT with raw variable IDs only
#   e5: additive compact DTT with a fixed shuffled semantic assignment
#
# Historical e1/e2 remain available for reproduction. The rejected combined e3
# candidate remains gated. This script always reuses scripts/train.py and never
# opens locked test data.

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-2026 2027 2028}"
CANDIDATES="${CANDIDATES:-e4 e5}"
BASE_CONFIG="${BASE_CONFIG:-configs/wadi_qwen3_06b.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SEMANTIC_SHUFFLE_SEED="${SEMANTIC_SHUFFLE_SEED:-2026}"
SYNTHETIC_SAMPLES="${SYNTHETIC_SAMPLES:-170}"
SYNTHETIC_DATA_DIR="${SYNTHETIC_DATA_DIR:-$PROJECT_ROOT/synthetic_data/WADI-CLEAN_X_train_full_l64_seed2026}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/cloud_logs/classifier_candidates_$RUN_TAG}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_COMBINED_CANDIDATE="${ALLOW_COMBINED_CANDIDATE:-0}"

mkdir -p "$LOG_ROOT"

candidate_slug() {
  case "$1" in
    e1) echo "e1_numeric_l64_l12_direct_head_ce_synrot${SYNTHETIC_SAMPLES}" ;;
    e2) echo "e2_dtt_add_l64_l12_direct_head_ce_nosynth" ;;
    e3) echo "e3_dtt_add_l64_l12_direct_head_ce_synrot${SYNTHETIC_SAMPLES}" ;;
    e4) echo "e4_dtt_add_idonly_l64_l12_direct_head_ce_nosynth" ;;
    e5) echo "e5_dtt_add_shuffled_s${SEMANTIC_SHUFFLE_SEED}_l64_l12_direct_head_ce_nosynth" ;;
    *)
      echo "Unknown candidate '$1'; expected e1, e2, e3, e4, or e5." >&2
      return 1
      ;;
  esac
}

candidate_description() {
  case "$1" in
    e1) echo "80 numeric tokens + classifier CE + synthetic anomalies" ;;
    e2) echo "80 numeric tokens + additive compact DTT + correct semantics + classifier CE, no synthetic" ;;
    e3) echo "80 numeric tokens + additive compact DTT + classifier CE + synthetic anomalies" ;;
    e4) echo "80 numeric tokens + additive compact DTT + raw IDs only + classifier CE, no synthetic" ;;
    e5) echo "80 numeric tokens + additive compact DTT + fixed shuffled semantics + classifier CE, no synthetic" ;;
    *) return 1 ;;
  esac
}

candidate_uses_dtt() {
  [[ "$1" == "e2" || "$1" == "e3" || "$1" == "e4" || "$1" == "e5" ]]
}

candidate_uses_synthetic() {
  [[ "$1" == "e1" || "$1" == "e3" ]]
}

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

read -r -a seed_list <<< "$SEEDS"
read -r -a candidate_list <<< "$CANDIDATES"
if [[ ${#seed_list[@]} -eq 0 || ${#candidate_list[@]} -eq 0 ]]; then
  echo "SEEDS and CANDIDATES must each contain at least one value." >&2
  exit 1
fi

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ || ! "$EVAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE and EVAL_BATCH_SIZE must be positive integers." >&2
  exit 1
fi
if [[ ! "$SEMANTIC_SHUFFLE_SEED" =~ ^[0-9]+$ ]]; then
  echo "SEMANTIC_SHUFFLE_SEED must be a non-negative integer." >&2
  exit 1
fi

if [[ ! -f "$PROJECT_ROOT/$BASE_CONFIG" ]]; then
  echo "Base config is missing: $PROJECT_ROOT/$BASE_CONFIG" >&2
  exit 1
fi

needs_synthetic=0
for candidate in "${candidate_list[@]}"; do
  candidate_slug "$candidate" >/dev/null
  if [[ "$candidate" == "e3" && "$ALLOW_COMBINED_CANDIDATE" != "1" ]]; then
    echo "Candidate e3 is gated until e1 and e2 have each been reviewed." >&2
    echo "After that review, rerun with ALLOW_COMBINED_CANDIDATE=1 CANDIDATES=e3." >&2
    exit 1
  fi
  if candidate_uses_synthetic "$candidate"; then
    needs_synthetic=1
  fi
done

if [[ "$needs_synthetic" == "1" ]]; then
  if [[ ! "$SYNTHETIC_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "SYNTHETIC_SAMPLES must be a positive integer." >&2
    exit 1
  fi
  for file_name in "${required_package_files[@]}"; do
    if [[ ! -f "$SYNTHETIC_DATA_DIR/$file_name" ]]; then
      echo "Synthetic package is incomplete: $SYNTHETIC_DATA_DIR/$file_name" >&2
      exit 1
    fi
  done
fi

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
echo "Base config: $BASE_CONFIG"
echo "Physical/evaluation batch: $BATCH_SIZE/$EVAL_BATCH_SIZE"
if [[ " ${candidate_list[*]} " == *" e5 "* ]]; then
  echo "Fixed semantic shuffle seed: $SEMANTIC_SHUFFLE_SEED"
fi
if [[ "$needs_synthetic" == "1" ]]; then
  echo "Synthetic package: $SYNTHETIC_DATA_DIR"
  echo "Synthetic windows per epoch: $SYNTHETIC_SAMPLES"
fi
echo "Logs: $LOG_ROOT"
echo "Dry run: $DRY_RUN"

# Seed is the outer loop so every completed group contains paired candidate
# results with identical seed-dependent endpoint sampling.
for seed in "${seed_list[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 1
  fi
  for candidate in "${candidate_list[@]}"; do
    slug="$(candidate_slug "$candidate")"
    description="$(candidate_description "$candidate")"
    run_name="${slug}_seed${seed}_${RUN_TAG}"
    output_dir="$PROJECT_ROOT/outputs/$run_name"
    log_file="$LOG_ROOT/$run_name.log"

    if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to reuse non-empty output directory: $output_dir" >&2
      echo "Choose a new RUN_TAG or remove only the explicitly reviewed failed run." >&2
      exit 1
    fi

    command=(
      "$PYTHON_BIN" -u scripts/train.py
      --config "$BASE_CONFIG"
      --run-name "$run_name"
      --seed "$seed"
      --window-length 64
      --moirai-layer 12
      --projector direct
      --classifier-loss-weight 1
      --vocab-loss-weight 0
      --batch-size "$BATCH_SIZE"
      --eval-batch-size "$EVAL_BATCH_SIZE"
      --prompt-variant process
      --full-run-authorized
      --device "$DEVICE"
    )

    if candidate_uses_dtt "$candidate"; then
      command+=(
        --discrete-to-text
        --dtt-semantic-style compact
        --dtt-numeric-mode all_active
      )
      case "$candidate" in
        e4)
          command+=(--dtt-semantic-variant id_only)
          ;;
        e5)
          command+=(
            --dtt-semantic-variant shuffled
            --dtt-semantic-shuffle-seed "$SEMANTIC_SHUFFLE_SEED"
          )
          ;;
        *)
          command+=(--dtt-semantic-variant correct)
          ;;
      esac
    fi
    if candidate_uses_synthetic "$candidate"; then
      command+=(
        --use-synthetic-anomalies
        --synthetic-data-dir "$SYNTHETIC_DATA_DIR"
        --synthetic-samples "$SYNTHETIC_SAMPLES"
        --synthetic-sampling epoch_stratified
      )
    fi

    echo
    echo "Starting candidate=$candidate seed=$seed run=$run_name"
    echo "Definition: $description"
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
