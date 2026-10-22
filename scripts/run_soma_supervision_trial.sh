#!/usr/bin/env bash
set -euo pipefail
phase=${1:?Expected smoke or pilot}
case "$phase" in smoke|pilot|decoder_smoke|decoder_pilot|observation_smoke|observation_pilot) ;; *) exit 2 ;; esac
tag=${2:-$phase}
[[ "$tag" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
cd ${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks
export PYTHONPATH="$PWD:${PRIVATE_STORAGE}/motionbricks-preparation/kimodo"
export OMP_NUM_THREADS=4
python=${PRIVATE_STORAGE}/motionbricks-preparation/envs/mb-bones/bin/python
output="out/soma_supervision_20261006/$tag"
[[ ! -e "$output" ]] || { printf 'Refusing existing output\n' >&2; exit 4; }
mkdir -p "$output"
nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits > "$output/gpu_at_launch.csv"
gpus=(GPU-9e67aa3f-f9e8-8ec0-029e-13d6345fa160 GPU-6309e70e-16a8-90a3-ea26-70d530098acc)
for gpu in "${gpus[@]}"; do
    row=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    memory=${row%%,*}; utilization=${row##*,}
    if (( memory > 1000 || utilization > 5 )); then
        printf 'GPU no longer idle: %s %s\n' "$gpu" "$row" >&2
        exit 3
    fi
done
steps=1000; extra=(--pilot)
if [[ "$phase" == *smoke ]]; then steps=2; extra=(--smoke-test); fi
dataset=${PRIVATE_STORAGE}/motionbricks-bones/pilot-v7
bundle=out/soma_control_v3/baseline_backend.pt
config=${PRIVATE_STORAGE}/motionbricks-preparation/runs/locomotion-consistency-20261005/control/upstream_config_overrides.yaml
policies=(legacy_splice source_v2)
if [[ "$phase" == decoder_* ]]; then policies=(frozen_decoder train_decoder); fi
if [[ "$phase" == observation_* ]]; then policies=(predicted_features executed_fk); fi
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
for index in 0 1; do
    policy=${policies[index]}
    (
        export CUDA_VISIBLE_DEVICES=${gpus[index]}
        supervision=$policy
        decoder_args=()
        if [[ "$phase" == decoder_* ]]; then supervision=source_v2; fi
        if [[ "$policy" == train_decoder ]]; then decoder_args=(--train-decoder); fi
        if [[ "$phase" == observation_* ]]; then supervision=source_v2; decoder_args=(--history-geometry "$policy"); fi
        "$python" -u scripts/train_soma_pose_rollout.py --dataset "$dataset" \
            --labels out/sole_region_repair_v3/labels_train --bundle "$bundle" \
            --tokenizer-config "$config" --output "$output/$policy" --arm B \
            --coupled-root-trial --supervision-policy "$supervision" --steps "$steps" "${extra[@]}" "${decoder_args[@]}"
        if [[ "$phase" == *pilot ]]; then
            "$python" -u scripts/evaluate_soma_decoder_refinement.py --dataset "$dataset" \
                --bundle "$output/$policy/backend.pt" --commit-frames 8 --output "$output/review_$policy"
            if [[ "$policy" == legacy_splice || "$policy" == frozen_decoder || "$policy" == predicted_features ]]; then
                "$python" -u scripts/evaluate_soma_decoder_refinement.py --dataset "$dataset" \
                    --bundle "$bundle" --commit-frames 8 --output "$output/review_original"
            fi
        fi
    ) > "$output/$policy.log" 2>&1 &
    pids+=("$!")
done
status=0
for index in 0 1; do
    if wait "${pids[index]}"; then printf '%s completed\n' "${policies[index]}";
    else printf '%s failed\n' "${policies[index]}" >&2; status=1; fi
done
exit "$status"
