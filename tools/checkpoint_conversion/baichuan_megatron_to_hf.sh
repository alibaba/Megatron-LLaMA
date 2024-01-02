python tools/checkpoint_conversion/baichuan_checkpoint_conversion.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path "PATH_TO_CHECKPOINT_GENERATED_BY_THIS_REPO" \
--save_path "PATH_TO_SAVE_CONVERTED_CHECKPOINT" \
--target_params_dtype "fp16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "PATH_TO_MEGATRON_SOURCE_CODE"
