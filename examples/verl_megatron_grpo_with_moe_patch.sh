#!/bin/bash
set -x

# run 
# cd /aistudio/workspace/huilian_ssd/moe_patch/examples && bash verl_megatron_grpo_with_moe_patch.sh

# ==============================================================================
# 1. 基础环境与路径配置
# ==============================================================================
# 初始化 cuda nccl 环境
source /aistudio/workspace/huilian_ssd/moe_experi/env/source_cuda_nccl.sh


# 环境变量 (严格保留原始参数)
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA="=mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1"
export NCCL_COLLNET_ENABLE=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_TIMEOUT=300
export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
# Enable FlashInfer sampler for better performance (requires flashinfer-python>=0.2.3)
# Uncomment the following line after installing flashinfer-python:
# export VLLM_USE_FLASHINFER_SAMPLER=1
export GLOO_SOCKET_IFNAME=eth0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# 优化显存碎片
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN NVTE_FLASH_ATTN

# ==============================================================================
# 2. 核心并行与 Batch 变量 (严格保留原始逻辑)
# ==============================================================================
NODES=2
PP=2
VPP=None
TP=2
EP=4
ETP=1
VLLM_INFER_TP=2

offload=False
gpu_memory_utilization=0.6
rollout_temperature=1
bs=32
ppo_mini_batch_size=32
micro_bs=32
rollout_n=16
use_dynamic_bsz=True
calculate_log_probs=True             # 是否计算rollout 采样seq概率

loss_mode=vanilla          # grpo默认用vanilla, loss_mode=gspo
bypass_mode=False          # True: rollout correction, False: calc old policy prob
use_policy_gradient=False  # True: use policy gradient no clip，False: PPO clip
use_kl_loss=False           # True: use KL loss, default kl_loss_coef = 0.001, policy_loss = pg_loss + kl_loss_coef * kl_loss

# DAPO的数据集需要这么长
max_prompt_length=2048
max_response_length=30720

actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2 ))
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2 ))

HF_MODEL_PATH="/aistudio/workspace/yangbolan003/origin_models/Qwen3/Qwen3-30B-A3B-Base"
TRAIN_DATA_PATH="/aistudio/workspace/yangbolan003/data/OpenR1-Math-220k/rl/train_4000.parquet"
# TEST_DATA_PATH="/aistudio/workspace/yangbolan003/data/OpenR1-Math-220k/rl/test_100.parquet"
TEST_DATA_PATH="/aistudio/workspace/yangbolan003/data/aime25/test.parquet" 

#==============================路径设置=====================================
# routing replay 配置
ROUTING_REPLAY_MODE="disabled"        # "R2" "disabled" "R3"  

# wandb
# 以下设置仅在单节点时有效果，多节点跑RLR，需要在任务界面加环境变量
export WANDB_API_KEY=local-248b3143cc0b83a1fc25f2e9cf93432c880ee8f7
export WANDB_BASE_URL=http://10.203.19.52:8006

project_name="moe_grpo_openr1Math_test"
exper_name=bs${bs}P${PP}T${TP}minbs${ppo_mini_batch_size}microbs${micro_bs}_r${rollout_n}_${ROUTING_REPLAY_MODE}_lr1e5
out_dir="/aistudio/workspace/yangbolan003/outputs/rl/$project_name/$exper_name/$(date +%Y%m%d%H%M)"

# routed expert load results dir
export MOE_PATCH_DIR="$out_dir/expert_load_results"
export ckpts_dir="$out_dir/models"
export rollout_data_dir="$out_dir/rollout_data"
export validation_data_dir="$out_dir/validation_data"
# 训练前手动预创建目录以防 FileNotFoundError
mkdir -p "$MOE_PATCH_DIR" "$ckpts_dir" "$rollout_data_dir" "$validation_data_dir"
# ==============================================================================
# 3. 参数化数组 (保持原代码所有 Key-Value 对)
# ==============================================================================

DATA=(
    data.train_files=$TRAIN_DATA_PATH
    data.val_files=$TEST_DATA_PATH
    data.prompt_key=prompt
    data.train_batch_size=$bs
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.filter_overlong_prompts=True
    data.truncation='error'
)

ACTOR_MEGATRON=(
    actor_rollout_ref.model.path=$HF_MODEL_PATH
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.router_replay.mode=${ROUTING_REPLAY_MODE}
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=8
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.optim.total_training_steps=125
    actor_rollout_ref.actor.optim.lr=3e-6
    actor_rollout_ref.actor.optim.lr_decay_style=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05
    actor_rollout_ref.actor.optim.min_lr=1e-6
    # Megatron uses "adam" as optimizer name, not "adamw"
    # To enable AdamW behavior (decoupled weight decay), set decoupled_weight_decay=True
    # Note: decoupled_weight_decay defaults to True in Megatron, so "adam" already behaves as AdamW
    actor_rollout_ref.actor.optim.optimizer=adam
    # Explicitly enable decoupled weight decay (AdamW mode) - optional, defaults to True
    # actor_rollout_ref.actor.optim.decoupled_weight_decay=True
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_bs
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size
    actor_rollout_ref.actor.checkpoint.save_contents=['model']
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode
)

ROLLOUT_REF=(
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_bs
    actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_INFER_TP
    # actor_rollout_ref.rollout.data_parallel_size=$VLLM_INFER_DP
    actor_rollout_ref.rollout.calculate_log_probs=$calculate_log_probs
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization
    actor_rollout_ref.rollout.n=$rollout_n
    actor_rollout_ref.rollout.temperature=$rollout_temperature
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    # Validation kwargs for pass@k evaluation
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7
    actor_rollout_ref.rollout.val_kwargs.n=16
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_bs
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP
    actor_rollout_ref.ref.megatron.param_offload=${offload}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
)

# patch 代码设置（需要在 TRAINER_CONFIG 之前定义，因为 TRAINER_CONFIG 会使用这些变量）
# export VERL_PATCH_PATH="/aistudio/workspace/huilian_ssd/moe_patch/src/patch/verl"
export VERL_PATCH_PATH="/aistudio/workspace/huilian_ssd/moe_experi/code/verl_patched"
export RAY_PYTHONPATH="${VERL_PATCH_PATH}:${PYTHONPATH:-}"

# 自定义 reward 函数路径
CUSTOM_REWARD_FUNCTION_PATH="/aistudio/workspace/huilian_ssd/moe_experi/code/verl_patched/custom_compute_score.py"

TRAINER_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=$bypass_mode
    # algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.use_policy_gradient=$use_policy_gradient
    trainer.default_local_dir=$ckpts_dir
    trainer.logger=['console','wandb']
    trainer.project_name=$project_name
    trainer.experiment_name="$exper_name"
    trainer.nnodes=$NODES
    trainer.n_gpus_per_node=8
    trainer.save_freq=125
    trainer.test_freq=10
    trainer.total_epochs=1
    # trainer.total_training_steps=
    trainer.resume_mode=disable
    trainer.rollout_data_dir=$rollout_data_dir
    +trainer.validation_data_dir=$validation_data_dir
    trainer.log_val_generations=True
    # 自定义 reward 函数配置
    # custom_reward_function.path=$CUSTOM_REWARD_FUNCTION_PATH
    # custom_reward_function.name=compute_score
)

RUN_TIME_CONFIG=(
    # Ray 将这个本地目录打包并分发到所有从节点的 sys.path 中
    "+ray_kwargs.ray_init.runtime_env.py_modules=['$VERL_PATCH_PATH']"
    # 设置环境变量（Ray 要求所有 env_vars 的值必须是字符串类型）
    "+ray_kwargs.ray_init.runtime_env.env_vars.MOE_PATCH_DIR=$MOE_PATCH_DIR"
    # 在megatron_worker.py 的底部，import VERL_APPLY_PATCHES，patch代码生效
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_APPLY_PATCHES=actor_routed_expert_capturer"
    # 确保 PYTHONPATH 包含 Patch 路径，且排在最前面以覆盖镜像原有的 verl
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$RAY_PYTHONPATH:\$PYTHONPATH'"
    # Ray 会在启动每个 Worker 进程后，自动运行 patcher.run_patch()，零代码入侵，但会提前分配显存，后续出现CUDA OOM 问题
    # "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook='patcher.run_patch'"
)
# 主进程也设置 PYTHONPATH
export PYTHONPATH="$RAY_PYTHONPATH"

# ==============================================================================
# 4. 执行 (保持 verl 启动脚本调用)
# ==============================================================================


python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ACTOR_MEGATRON[@]}" \
    "${ROLLOUT_REF[@]}" \
    "${RUN_TIME_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "${TRANSFER_QUEUE_CONFIG[@]}" \
    trainer.balance_batch=False \
    trainer.val_before_train=True 2>&1


# 画热力图
# verl_moe_patch_dir="/aistudio/workspace/yangbolan003/experiments/outputs/rl/moe_grpo_openr1Math_test/bs32P2T2minbs32microbs32_r16_disabled_lr1e5/202602061044/expert_load_results"

# python /aistudio/workspace/huilian_ssh/moe_experi/code/swift/moe_monitor_patch/plot_moe_heatmap.py $verl_moe_patch_dir \
#     --iter 3 \
#     --layers 6 12 \
#     --out ./verl_moe_ld_step3_layer6_12.png
