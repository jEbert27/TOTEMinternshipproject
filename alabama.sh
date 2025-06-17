#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""

seq_len=104
root_path_name="$(pwd)/data/al"
data_path_name="flu_al.csv"
data_name="alabama"
random_seed=47
pred_len=4

# 1) RevIN-normalize & split (writes into forecasting/data/alabama/…)
python -u forecasting/save_revin_data.py \
  --random_seed "${random_seed}" \
  --data "${data_name}" \
  --root_path "${root_path_name}" \
  --data_path "${data_path_name}" \
  --features M \
  --seq_len "${seq_len}" \
  --pred_len "${pred_len}" \
  --save_path "forecasting/data/${data_name}"

# 2) Train VQ-VAE → produces forecasting/data/alabama/Tin104_Tout4/codebook.npy
python forecasting/train_vqvae.py \
  --config_path forecasting/scripts/alabama.json \
  --model_init_num_gpus 0 \
  --data_init_cpu_or_gpu cpu \
  --comet_log false \
  --save_path "forecasting/saved_models/${data_name}" \
  --base_path "forecasting/data/${data_name}" \
  --batchsize 1024

# 3) Train & eval the 4-week forecaster
for Tout in ${pred_len}; do
  python forecasting/train_forecaster.py \
    --data-type "${data_name}" \
    --Tin "${seq_len}" \
    --Tout "${Tout}" \
    --seed "${random_seed}" \
    --data_path "forecasting/data/${data_name}/Tin${seq_len}_Tout${Tout}" \
    --codebook_size 256 \
    --checkpoint \
    --checkpoint_path "forecasting/saved_models/${data_name}/forecaster_checkpoints/${data_name}_Tin${seq_len}_Tout${Tout}_seed${random_seed}" \
    --file_save_path "forecasting/results/${data_name}/"
done

