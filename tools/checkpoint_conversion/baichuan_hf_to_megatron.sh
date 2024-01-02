python baichuan_checkpoint_conversion.py \
--load_path "PATH_TO_CHECKPOINT_DOWNLOADED_FROM_HUGGINGFACE" \
--save_path "PATH_TO_SAVE_CONVERTED_CHECKPOINT" \
--target_tensor_model_parallel_size 2 \
--target_pipeline_model_parallel_size 1 \
--target_data_parallel_size 16 \
--target_params_dtype "fp16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "PATH_TO_MEGATRON_SOURCE_CODE"
