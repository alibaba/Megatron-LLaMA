python baichuan_checkpoint_conversion.py \
--load_path "/data/share_user/quyincen/megatron_model/hf_model/60132_model" \
--save_path "/data/share_user/quyincen/megatron_model/mg_model" \
--target_tensor_model_parallel_size 2 \
--target_pipeline_model_parallel_size 1 \
--target_data_parallel_size 4 \
--target_params_dtype "bf16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "/data/share_user/quyincen/Megatron-LLaMA" \
