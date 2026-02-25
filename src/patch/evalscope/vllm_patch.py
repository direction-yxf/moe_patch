"""
自动记录 MoE 模型的 expert 使用统计
"""

import atexit
import json
import os
import threading
import time
from collections import defaultdict
import torch


class ExpertStatsRecorder:
    _instance = None
    
    EVALSCOPE_MOE_STATS_FILENAME = "evalscope_moe_lb_step_all_rank_all.jsonl"
    TOP_K = 8
    NUM_EXPERTS = 128

    @staticmethod
    def _layer_idx(layer_key: str) -> int:
        import re
        m = re.search(r'\.(\d+)\.', layer_key)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _layer_name(layer_key: str) -> str:
        import re
        m = re.search(r'\.(\d+)\.', layer_key)
        return f"layer_{m.group(1)}" if m else layer_key

    def __init__(self):
        moe_patch_dir = os.environ.get('MOE_PATCH_DIR', '')
        self.output_file = os.path.join(moe_patch_dir, self.EVALSCOPE_MOE_STATS_FILENAME) if moe_patch_dir else ''
        self.enabled = bool(self.output_file)
        self.counts = defaultdict(lambda: defaultdict(int))
        self._total_count = 0

        print(f"[ExpertStatsRecorder] Initialized: enabled={self.enabled}, output_file={self.output_file}")
        
        if self.enabled and self.output_file:
            atexit.register(self._auto_save)
            threading.Thread(target=self._loop_save, daemon=True).start()
            print(f"[ExpertStatsRecorder] Auto-save thread started")

    def _loop_save(self):
        while True:
            time.sleep(10)  # 每 60 秒执行一次
            try:
                self.save()
            except:
                pass

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _auto_save(self):
        if self.output_file and self.counts:
            try:
                print(f"[ExpertStatsRecorder] _auto_save triggered, counts has {len(self.counts)} layers")
                self.save(self.output_file)
            except Exception as e:
                print(f"[ExpertStatsRecorder] Error in _auto_save: {e}")
                import traceback
                traceback.print_exc()
    
    def record(self, layer_name: str, topk_ids: torch.Tensor):
        """
        core：记录专家使用统计
        """
        if not self.enabled:
            return
        expert_ids = topk_ids.flatten().cpu().tolist()
        for expert_id in expert_ids:
            self.counts[layer_name][expert_id] += 1
        self._total_count += len(expert_ids)
    
    def save(self, output_path: str = None):
        if not self.enabled:
            return
        if output_path is None:
            output_path = self.output_file
        if not output_path:
            return

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            for layer, experts in sorted(self.counts.items(), key=lambda x: self._layer_idx(x[0])):
                num_experts = max(self.NUM_EXPERTS, max((int(k) for k in experts), default=-1) + 1)
                actual_assignments = [experts.get(i, 0) for i in range(num_experts)]
                tokens = sum(actual_assignments) // self.TOP_K
                record = {
                    "iteration": 0,
                    "rank": 0,
                    "layer": self._layer_name(layer),
                    "num_experts": num_experts,
                    "top_k": self.TOP_K,
                    "tokens": tokens,
                    "actual_assignments": actual_assignments,
                }
                f.write(json.dumps(record) + '\n')


_recorder = None
_patched = False
_layer_name_map = {}  # 用于存储layer对象到层名的映射
_layer_counter = 0
_call_count = 0  # 用于调试


def record_expert_usage(layer_name: str, topk_ids: torch.Tensor):
    global _recorder
    if _recorder is None:
        _recorder = ExpertStatsRecorder.get_instance()
    _recorder.record(layer_name, topk_ids)


def apply():
    global _recorder, _patched
    
    print(f"[vllm_patch] apply() called, _patched={_patched}")
    
    if _patched:
        print(f"[vllm_patch] Already patched, returning")
        return True
    
    _recorder = ExpertStatsRecorder.get_instance()
    if not _recorder.enabled:
        print(f"[vllm_patch] Recorder not enabled, output_file={_recorder.output_file}")
        return False
    
    try:
        from vllm.model_executor.layers.fused_moe.layer import (
            UnquantizedFusedMoEMethod, FusedMoE
        )
        print(f"[vllm_patch] Successfully imported UnquantizedFusedMoEMethod and FusedMoE")
        
        if not hasattr(UnquantizedFusedMoEMethod, 'forward_cuda'):
            print(f"[vllm_patch] UnquantizedFusedMoEMethod has no forward_cuda method")
            return False
        
        original_forward_cuda = UnquantizedFusedMoEMethod.forward_cuda
        print(f"[vllm_patch] Got original_forward_cuda, starting to patch...")
        
        def patched_forward_cuda(self, layer, x, use_grouped_topk, top_k,
                                router_logits, renormalize, topk_group=None,
                                num_expert_group=None, global_num_experts=-1,
                                expert_map=None, custom_routing_function=None,
                                scoring_func="softmax", e_score_correction_bias=None,
                                apply_router_weight_on_input=False, activation="silu",
                                enable_eplb=False, expert_load_view=None,
                                logical_to_physical_map=None, logical_replica_count=None):
            
            topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                indices_type=self.topk_indices_dtype,
                enable_eplb=enable_eplb,
                expert_map=expert_map,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count
            )
            
            # 核心patch - 获取层名
            global _layer_name_map, _layer_counter, _call_count, _recorder
            _call_count += 1
            
            if layer not in _layer_name_map:
                if hasattr(layer, 'layer_name'):
                    _layer_name_map[layer] = layer.layer_name
                    print(f"[vllm_patch] New layer found with layer_name: {layer.layer_name}")
                elif hasattr(layer, 'name') and layer.name:
                    _layer_name_map[layer] = layer.name
                    print(f"[vllm_patch] New layer found with name: {layer.name}")
                else:
                    # 使用全局计数器分配稳定的层名
                    _layer_name_map[layer] = f"layer_{_layer_counter}"
                    print(f"[vllm_patch] New layer found, assigned name: layer_{_layer_counter}")
                    _layer_counter += 1
            
            layer_name = _layer_name_map[layer]
            # 调试：前几次调用打印详细信息
            if _call_count <= 10:
                expert_ids = topk_ids.flatten().cpu().tolist()[:5]  # 只打印前5个
                print(f"[vllm_patch] patched_forward_cuda called (call #{_call_count}): layer={layer_name}, topk_ids sample={expert_ids}, shape={topk_ids.shape}")
                if _recorder and _recorder.counts:
                    print(f"[vllm_patch] Current _recorder.counts: {dict(_recorder.counts)}")
            
            if _recorder is None:
                _recorder = ExpertStatsRecorder.get_instance()
            record_expert_usage(layer_name, topk_ids)
            
            if self.rocm_aiter_moe_enabled:
                return self.rocm_aiter_fused_experts(
                    hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
                    topk_weights=topk_weights, topk_ids=topk_ids,
                    expert_map=expert_map, activation=activation,
                    apply_router_weight_on_input=apply_router_weight_on_input
                )
            elif self.fused_experts is not None:
                if self.has_bias:
                    raise ValueError("FusedMoEModularKernel does not support bias.")
                return self.fused_experts(
                    hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
                    topk_weights=topk_weights, topk_ids=topk_ids,
                    inplace=True, activation=activation,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    global_num_experts=global_num_experts, expert_map=expert_map,
                )
            else:
                return original_forward_cuda(
                    self, layer, x, use_grouped_topk, top_k, router_logits,
                    renormalize, topk_group, num_expert_group, global_num_experts,
                    expert_map, custom_routing_function, scoring_func,
                    e_score_correction_bias, apply_router_weight_on_input,
                    activation, enable_eplb, expert_load_view,
                    logical_to_physical_map, logical_replica_count
                )
        
        UnquantizedFusedMoEMethod.forward_cuda = patched_forward_cuda
        _patched = True
        print(f"[vllm_patch] Patch applied successfully!")
        return True
        
    except Exception as e:
        print(f"[vllm_patch] Failed to apply patch: {e}")
        import traceback
        traceback.print_exc()
        return False


# 自动应用（在子进程中）
if os.environ.get('MOE_PATCH_DIR'):
    try:
        result = apply()
        print(f"[vllm_patch] Auto-apply result: {result}")
    except Exception as e:
        print(f"[vllm_patch] Auto-apply failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[vllm_patch] MOE_PATCH_DIR not set, skipping auto-apply")
