# MOE-Patch 🔍 

**MOE-Patch** 是一个专为 MoE 模型设计的深度监测工具。它通过 **Monkey Patch** 机制，在不侵入原始训练或推理框架（如 `verl`, `ms-swift` , `vllm`等）源码的前提下，实时捕获并分析路由分布、专家负载、Token 丢弃率等关键指标。

## 📝 更新日志

[v0.1.0] - 2025-02-03 (v1版本)**"v1版本发布"**


## ✨ 核心特性

- **即插即用**：基于 Monkey Patch 技术，适配LLM训练和推理流程。
- **多维指标**：
    - **Load Balance**: 相对负载均衡
    - **Importance Sampling weight**: 重要性采样权重
- **可视化**：支持绘制负载均衡热力分布

## 🚀 快速上手

### 统计
无需修改直接带着patch代码训练或者推理
```bash
#routed expert record dir
export MOE_PATCH_DIR="path/to/moe_patch_dir"     
#patched code dir
export VERL_PATCH_PATH="path/to/moe_patch/src/patch/verl"  
export RAY_PYTHONPATH="${VERL_PATCH_PATH}:${PYTHONPATH:-}"
export PYTHONPATH=$RAY_PYTHONPATH                         
export VERL_APPLY_PATCHES=actor_routed_expert_capturer_v3  # apply 的 补丁文件

### 多机
RUN_TIME_CONFIG=(
    "+ray_kwargs.ray_init.runtime_env.py_modules=['$VERL_PATCH_PATH']"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MOE_PATCH_DIR=$MOE_PATCH_DIR"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_APPLY_PATCHES=actor_routed_expert_capturer_v3"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$RAY_PYTHONPATH:\$PYTHONPATH'"
)

python3 -m verl.trainer.main_ppo \ 
    xx
    "${RUN_TIME_CONFIG[@]}" \
    xx
```

```python
# 在 https://github.com/verl-project/verl/blob/main/verl/workers/megatron_workers.py ,末尾添加


# 在 https://github.com/verl-project/verl/blob/e5f5ea6620dbe0409617fe1643f390974017bb87/verl/trainer/ppo/ray_trainer.py#L1605 后添加

```


### 可视化
```bash
python src/visual_moe_patch.py 统计.jsonl
```

![verl_moe_ld_step3_layer6_12](assets/verl_moe_ld_step3_layer6_12.png)


## 🤝 贡献与反馈
如果你在特定的 MoE 架构上遇到适配问题，欢迎提交 [Issue](https://github.com/your-username/expertlens/issues)。

## 📄 开源协议
本项目基于 [Apache-2.0](LICENSE) 协议开源。