"""
自动记录 MoE 模型的 expert 使用统计
"""

import atexit
import json
import os
from collections import defaultdict
import torch


class ExpertStatsRecorder:
    _instance = None
    
    def __init__(self):
        self.output_file = os.environ.get('VLLM_EXPERT_STATS_FILE', '')
        self.enabled = bool(self.output_file)
        self.counts = defaultdict(lambda: defaultdict(int))
        
        if self.enabled and self.output_file:
            atexit.register(self._auto_save)
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _auto_save(self):
        if self.output_file and self.counts:
            try:
                self.save(self.output_file)
            except:
                pass
    
    def record(self, layer_name: str, topk_ids: torch.Tensor):
        """
        core：记录专家使用统计
        """
        if not self.enabled:
            return
        expert_ids = topk_ids.flatten().cpu().tolist()
        for expert_id in expert_ids:
            self.counts[layer_name][expert_id] += 1
    
    def save(self, output_path: str = None):
        if not self.enabled:
            return
        if output_path is None:
            output_path = self.output_file
        if not output_path:
            return
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        output_data = {
            "counts": {
                layer: dict(experts) 
                for layer, experts in self.counts.items()
            }
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)


_recorder = None
_patched = False


def record_expert_usage(layer_name: str, topk_ids: torch.Tensor):
    global _recorder
    if _recorder is None:
        _recorder = ExpertStatsRecorder.get_instance()
    _recorder.record(layer_name, topk_ids)


def apply():
    global _recorder, _patched
    
    if _patched:
        return True
    
    _recorder = ExpertStatsRecorder.get_instance()
    if not _recorder.enabled:
        return False
    
    try:
        from vllm.model_executor.layers.fused_moe.layer import (
            UnquantizedFusedMoEMethod, FusedMoE
        )
        
        if not hasattr(UnquantizedFusedMoEMethod, 'forward_cuda'):
            return False
        
        original_forward_cuda = UnquantizedFusedMoEMethod.forward_cuda
        
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
            
            # 核心patch
            if hasattr(layer, 'layer_name'):
                record_expert_usage(layer.layer_name, topk_ids)
            
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
        return True
        
    except:
        return False


# 自动应用（在子进程中）
if os.environ.get('VLLM_EXPERT_STATS_FILE'):
    try:
        apply()
    except:
        pass
