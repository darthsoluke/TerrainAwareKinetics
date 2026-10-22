#!/usr/bin/env bash
set -euo pipefail
phase=${1:?Expected smoke or full}
case "$phase" in smoke|full) ;; *) exit 2 ;; esac
tag=${2:-$phase}
if [[ ! "$tag" =~ ^[a-zA-Z0-9_]+$ ]]; then exit 2; fi
cd ${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks
export PYTHONPATH="$PWD:${PRIVATE_STORAGE}/motionbricks-preparation/kimodo"
export OMP_NUM_THREADS=4
python=${PRIVATE_STORAGE}/motionbricks-preparation/envs/mb-bones/bin/python
output=out/soma_joint_rollout_20261006
dataset=${PRIVATE_STORAGE}/motionbricks-bones/pilot-v7
bundle=out/soma_control_v3/baseline_backend.pt
config=${PRIVATE_STORAGE}/motionbricks-preparation/runs/locomotion-consistency-20261005/control/upstream_config_overrides.yaml
gpus=(GPU-6309e70e-16a8-90a3-ea26-70d530098acc GPU-7ccaf2e9-8580-9c1c-ebd9-7c34f51ad99e)
mkdir -p "$output"
nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits > "$output/${tag}_gpu_snapshot.csv"
for gpu in "${gpus[@]}"; do
    row=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    memory=${row%%,*}; utilization=${row##*,}
    if (( memory > 1000 || utilization > 5 )); then
        printf 'GPU not idle: %s %s\n' "$gpu" "$row" >&2
        exit 3
    fi
done
steps=10000
extra=()
if [[ "$phase" == smoke ]]; then steps=2; extra=(--smoke-test); fi
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
for index in 0 1; do
    arm=A; if (( index == 1 )); then arm=B; fi
    destination="$output/${tag}_${arm}"
    if [[ -e "$destination" ]]; then printf 'Refusing existing output %s\n' "$destination" >&2; exit 4; fi
done
for index in 0 1; do
    arm=A; if (( index == 1 )); then arm=B; fi
    (
        export CUDA_VISIBLE_DEVICES=${gpus[index]}
        "$python" -u scripts/train_soma_pose_rollout.py --dataset "$dataset" \
            --labels out/sole_region_repair_v3/labels_train --bundle "$bundle" \
            --tokenizer-config "$config" --output "$output/${tag}_${arm}" \
            --arm "$arm" --coupled-root-trial --steps "$steps" "${extra[@]}"
        if [[ "$phase" == full ]]; then
            "$python" -u scripts/evaluate_soma_decoder_refinement.py --dataset "$dataset" \
                --bundle "$output/${tag}_${arm}/backend.pt" --commit-frames 8 \
                --output "$output/review_${arm}"
            if [[ "$arm" == A ]]; then
                "$python" -u scripts/evaluate_soma_decoder_refinement.py --dataset "$dataset" \
                    --bundle "$bundle" --commit-frames 8 --output "$output/review_Original"
            fi
        fi
    ) > "$output/${tag}_${arm}.log" 2>&1 &
    pids+=("$!")
done
status=0
for index in 0 1; do
    if wait "${pids[index]}"; then
        printf '%s arm %s completed\n' "$phase" "$index"
    else
        result=$?
        printf '%s arm %s failed: %s\n' "$phase" "$index" "$result" >&2
        status=1
    fi
done
exit "$status"
