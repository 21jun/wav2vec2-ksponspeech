#!/usr/bin/env bash
python3 run_asr_roman_new_token.py \
--output_dir="./wav2vec2-base-100h-roman-new-token" \
--num_train_epochs="30" \
--per_device_train_batch_size="8" \
--per_device_eval_batch_size="8" \
--evaluation_strategy="steps" \
--save_total_limit="3" \
--save_steps="300" \
--eval_steps="300" \
--logging_steps="20" \
--learning_rate="5e-4" \
--warmup_steps="3000" \
--model_name_or_path="facebook/wav2vec2-base" \
--fp16 \
--preprocessing_num_workers="8" \
--group_by_length \
--freeze_feature_extractor
