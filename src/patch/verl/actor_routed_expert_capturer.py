
"""
Expert Load Statistics Capturer for MegatronPPOActor - V3 Production (Optimized)
主要改进：
1. 异步保存功能（AsyncSaveManager）：减少训练过程中的I/O阻塞
2. 简化的 layer_name 格式：直接使用 "layer_0" 而不是 "model.layers.0.mlp.experts"
3. 完整的 disabled 模式支持：使用 response_mask 正确过滤 padding tokens
4. R2/R3 模式支持：在 loss_func 中收集统计
5. Per-DP 保存：每个 Data Parallel rank 保存自己的数据
6. Pipeline Parallelism 支持：使用 all_gather_object 正确收集跨 PP ranks 的统计
7. 用户配置支持：enable_routing_replay 字段自动设置
8. 文件锁保护：使用 fcntl.flock 防止并发写入冲突
9. ✨ Step级别累积（V3优化）：在内存中累积一个完整step的数据，只保存一次
   - 减少87.5%的文件I/O操作（从每个mini_batch保存 → 每个step保存）
   - 消除文件锁争用（不同step独立文件）
   - 避免重复读写（内存合并）
   - 预期节省训练时间：3-4小时（从12小时降到8-9小时）
"""

import os
import json
import time
import threading
from datetime import datetime
from functools import partial, wraps
from typing import Iterable, Any
from queue import Queue


# ================= 异步保存管理器 =================

class AsyncSaveManager:
    """异步保存管理器，使用单独的线程处理文件 I/O"""
    
    def __init__(self, max_workers=2):
        """
        Args:
            max_workers: 最大并发保存线程数（默认2个）
        """
        self.save_queue = Queue()
        self.workers = []
        self.should_stop = threading.Event()
        self.active_tasks = 0
        self.lock = threading.Lock()
        
        # 启动 worker 线程
        for i in range(max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncSaveWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"[AsyncSave] Initialized with {max_workers} worker threads")
    
    def _worker_loop(self):
        """Worker 线程的主循环"""
        while not self.should_stop.is_set():
            try:
                # 从队列获取任务（timeout 1秒）
                task = self.save_queue.get(timeout=1.0)
                if task is None:  # 停止信号
                    break
                
                # 执行保存任务
                save_func, args, kwargs = task
                try:
                    save_func(*args, **kwargs)
                except Exception as e:
                    print(f"[AsyncSave-ERROR] Save failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    with self.lock:
                        self.active_tasks -= 1
                    self.save_queue.task_done()
                    
            except Exception:
                # Queue.get timeout，继续循环
                continue
    
    def submit(self, save_func, *args, **kwargs):
        """提交一个保存任务
        
        Args:
            save_func: 保存函数
            *args, **kwargs: 传递给 save_func 的参数
        """
        with self.lock:
            self.active_tasks += 1
        
        self.save_queue.put((save_func, args, kwargs))
        print(f"[AsyncSave] Task submitted, queue size: {self.save_queue.qsize()}, active: {self.active_tasks}")
    
    def wait_all(self, timeout=300):
        """等待所有保存任务完成
        
        Args:
            timeout: 最大等待时间（秒），默认 300 秒
        
        Returns:
            bool: 是否所有任务都完成
        """
        print(f"[AsyncSave] Waiting for all tasks to complete (timeout={timeout}s)...")
        start_time = time.time()
        
        # 等待队列为空
        while not self.save_queue.empty() or self.active_tasks > 0:
            time.sleep(0.1)
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"[AsyncSave-WARN] Timeout after {timeout}s, {self.active_tasks} tasks still active")
                return False
        
        print(f"[AsyncSave] All tasks completed in {time.time() - start_time:.2f}s")
        return True
    
    def shutdown(self, wait=True, timeout=300):
        """关闭异步保存管理器
        
        Args:
            wait: 是否等待所有任务完成
            timeout: 最大等待时间
        """
        print(f"[AsyncSave] Shutting down...")
        
        if wait:
            self.wait_all(timeout=timeout)
        
        # 发送停止信号
        self.should_stop.set()
        for _ in self.workers:
            self.save_queue.put(None)
        
        # 等待 worker 线程结束
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        print(f"[AsyncSave] Shutdown complete")


def _collect_expert_stats_r2_mode(self, routed_experts_response, response_mask):
    """R2模式：从routed_experts收集expert统计（已过滤到response部分）
    
    Args:
        routed_experts_response: Response部分的routing tensor
                                Shape: (batch_size, response_len, num_layers, top_k)
        response_mask: Response mask tensor
                      Shape: (batch_size, response_len)
    
    Returns:
        dict: {layer_name: {expert_id_str: count}} (如 {"layer_0": {"0": 123, "1": 234}})
    """
    import torch
    
    collected_stats = {}
    
    # Shape: (batch_size, response_len, num_layers, top_k)
    batch_size, response_len, num_layers, top_k = routed_experts_response.shape
    
    print(f'[DEBUG-r2-collect] Processing batch_size={batch_size}, response_len={response_len}, layers={num_layers}')
    
    # 处理所有layers
    for global_layer_idx in range(num_layers):
        layer_routing = routed_experts_response[:, :, global_layer_idx, :]  # (batch, response_len, top_k)
        
        # 应用response_mask过滤
        mask_expanded = response_mask.unsqueeze(-1)  # (batch, response_len, 1)
        masked_routing = layer_routing[mask_expanded.expand_as(layer_routing)]
        valid_expert_indices = masked_routing.flatten()
        
        # Filter invalid
        if (valid_expert_indices < 0).any():
            valid_expert_indices = valid_expert_indices[valid_expert_indices >= 0]
        
        if valid_expert_indices.numel() == 0:
            continue
        
        # ✨ 直接使用 layer_name 格式
        layer_name = f"layer_{global_layer_idx}"
        if layer_name not in collected_stats:
            collected_stats[layer_name] = {}
        
        # Count
        unique_experts = valid_expert_indices.unique()
        for expert_id in unique_experts:
            count = (valid_expert_indices == expert_id).sum().item()
            expert_id_str = str(expert_id.item())
            if expert_id_str not in collected_stats[layer_name]:
                collected_stats[layer_name][expert_id_str] = 0
            collected_stats[layer_name][expert_id_str] += count
    
    return collected_stats


def _collect_and_accumulate_r2_stats(self, data, response_mask):
    """在R2/R3模式下收集并累积expert统计（在loss_func中调用）
    
    Args:
        data: DataProto，包含routed_experts数据
        response_mask: Response mask tensor，shape (batch_size, response_len)
    
    Returns:
        None (直接累积到self._cumulative_expert_stats)
    """
    routed_experts = data.get("routed_experts", None) if data is not None else None
    if routed_experts is None:
        return
    
    print('[DEBUG-loss_func] Found routed_experts in R2/R3 mode, collecting stats')
    print(f'[DEBUG-loss_func] routed_experts shape (full seq): {routed_experts.shape}')
    print(f'[DEBUG-loss_func] response_mask shape: {response_mask.shape}')
    
    # 我们只需要统计response部分的expert使用情况
    response_length = response_mask.size(1)
    print(f'[DEBUG-loss_func] response_length: {response_length}')
    
    # 从完整序列中提取response部分: [:, -response_length:, :, :]
    routed_experts_response = routed_experts[:, -response_length:, :, :]
    print(f'[DEBUG-loss_func] routed_experts_response shape: {routed_experts_response.shape}')
    
    # 传递response_mask用于进一步过滤
    current_rank_stats = self._collect_expert_stats_r2_mode(
        routed_experts_response, 
        response_mask=response_mask
    )
    
    # 直接累加到全局统计（routed_experts已经包含所有layers）
    for layer_name, expert_counts in current_rank_stats.items():
        if layer_name not in self._cumulative_expert_stats:
            self._cumulative_expert_stats[layer_name] = {}
        for expert_id_str, count in expert_counts.items():
            if expert_id_str not in self._cumulative_expert_stats[layer_name]:
                self._cumulative_expert_stats[layer_name][expert_id_str] = 0
            self._cumulative_expert_stats[layer_name][expert_id_str] += count
    
    print(f'[DEBUG-loss_func] Collected stats for {len(current_rank_stats)} layers (response only)')


def _collect_disabled_mode_expert_stats(self, current_step, n_micro_batch):
    """处理disabled模式的expert统计（应用response_mask过滤、pp_gather、收集）
    
    Args:
        current_step: 当前训练step
        n_micro_batch: micro_batch数量
    
    Returns:
        dict: {layer_name: {expert_id: count}} 统计数据（不再直接保存）
    
    注意：
        - response_mask在forward_step中收集（所有PP ranks都执行），所以所有ranks都有
        - 因此能够正确过滤所有layers的prompt tokens，得到完整48层的统计
        - 统计数据会在 forward_backward_batch_patch 中累积到内存
        - ✨ V3优化：不再立即保存，而是返回统计数据供累积器使用
    """
    import torch
    from verl.utils.megatron.router_replay_utils import pp_gather
    from megatron.core import parallel_state as mpu
    
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    
    print(f'[DEBUG-disabled] PP rank {pp_rank}: Processing disabled mode, n_micro_batch={n_micro_batch}')
    print(f'[DEBUG-disabled] _disabled_routed_experts_buffer length: {len(self._disabled_routed_experts_buffer)}')
    print(f'[DEBUG-disabled] _disabled_response_mask_buffer length: {len(self._disabled_response_mask_buffer) if hasattr(self, "_disabled_response_mask_buffer") else 0}')
    
    # 打印每个buffer元素的shape
    if hasattr(self, '_disabled_routed_experts_buffer') and self._disabled_routed_experts_buffer:
        for i, routed in enumerate(self._disabled_routed_experts_buffer):
            print(f'[DEBUG-disabled] _disabled_routed_experts_buffer[{i}].shape = {routed.shape}')
    if hasattr(self, '_disabled_response_mask_buffer') and self._disabled_response_mask_buffer:
        for i, mask in enumerate(self._disabled_response_mask_buffer):
            print(f'[DEBUG-disabled] _disabled_response_mask_buffer[{i}].shape = {mask.shape}')
    
    # 验证buffer - 但不要提前return，因为pp_gather需要所有ranks参与
    has_valid_data = True
    if not hasattr(self, '_disabled_response_mask_buffer') or not self._disabled_response_mask_buffer:
        print(f'[WARN-disabled] No response_mask_buffer found on PP rank {mpu.get_pipeline_model_parallel_rank()}')
        print(f'[WARN-disabled] This should NOT happen after fix - response_mask should be collected in forward_step (all ranks)')
        has_valid_data = False
    elif len(self._disabled_routed_experts_buffer) != len(self._disabled_response_mask_buffer):
        print(f'[ERROR] Buffer length mismatch on PP rank {mpu.get_pipeline_model_parallel_rank()}: routed={len(self._disabled_routed_experts_buffer)}, masks={len(self._disabled_response_mask_buffer)}')
        print(f'[ERROR] This should NOT happen after fix - both should be collected in forward_step (all ranks)')
        has_valid_data = False
    
    # 对每个micro_batch应用response_mask过滤，shape [num_valid_tokens, num_local_layers, top_k]
    filtered_micro_batches = []
    
    if has_valid_data:
        for micro_idx, (routed_data, response_mask) in enumerate(zip(
            self._disabled_routed_experts_buffer, 
            self._disabled_response_mask_buffer
        )):
            # routed_data shape: (batch_size, seq_len, num_local_layers, top_k)
            # response_mask shape: (batch_size, response_len)
            
            batch_size, seq_len, num_local_layers, top_k = routed_data.shape
            response_len = response_mask.size(1)
            
            print(f'[DEBUG-disabled-filter] Micro_batch {micro_idx}:')
            print(f'  routed_data.shape = {routed_data.shape}, device = {routed_data.device}')
            print(f'  response_mask.shape = {response_mask.shape}, device = {response_mask.device}')
            
            # 验证batch_size匹配
            if routed_data.shape[0] != response_mask.shape[0]:
                print(f'[ERROR] Batch size mismatch at micro_batch {micro_idx}: routed={routed_data.shape[0]}, mask={response_mask.shape[0]}')
                continue
            
            # 验证response_len不超过seq_len
            if response_len > seq_len:
                print(f'[ERROR] Response length {response_len} exceeds seq_len {seq_len} at micro_batch {micro_idx}')
                continue
            
            # 提取response部分（序列的最后response_len个tokens）
            routed_response = routed_data[:, -response_len:, :, :]  # (batch, response_len, layers, top_k)
            
            # 应用response_mask过滤
            routed_flat = routed_response.reshape(-1, num_local_layers, top_k)
            mask_flat = response_mask.reshape(-1).to(routed_flat.device)  # ✨ 确保mask和data在同一设备
            routed_filtered = routed_flat[mask_flat]  # (num_valid_tokens, layers, top_k)
            
            print(f'  Filtered: {routed_flat.shape[0]} → {routed_filtered.shape[0]} tokens ({routed_filtered.shape[0]/routed_flat.shape[0]*100:.1f}%)')
            
            filtered_micro_batches.append(routed_filtered)
    
    # 即使没有有效数据,，也要参与pp_gather（集体通信需要所有ranks），要返回空tensor
    if not filtered_micro_batches:
        print(f'[WARN-disabled] PP rank {pp_rank}: No valid filtered micro-batches, creating empty tensor for pp_gather')
        # 获取layer数量信息
        from verl.utils.megatron.router_replay_utils import get_current_rank_layer_info
        local_rank_info = get_current_rank_layer_info(self.tf_config, vp_rank=None)
        num_local_layers = local_rank_info["count"]
        top_k = 8  # 默认top_k值，从RouterReplay配置中获取
        
        # 创建空tensor：(0, num_local_layers, top_k)
        disabled_layers_top_k_idx = torch.zeros((0, num_local_layers, top_k), dtype=torch.uint8)
    else:
        # 合并所有micro_batches的filtered tokens
        disabled_layers_top_k_idx = torch.cat(filtered_micro_batches, dim=0)
        print(f'[DEBUG-disabled-filter] PP rank {pp_rank}: After filtering and concat: shape={disabled_layers_top_k_idx.shape}')
        disabled_layers_top_k_idx = disabled_layers_top_k_idx.to(torch.uint8)
    
    # 不在tensor级别gather（因为不同ranks的tokens数量可能不同）
    # 每个rank独立统计自己的local layers，然后使用all_gather_object收集统计结果
    
    total_tokens, num_local_layers, top_k = disabled_layers_top_k_idx.shape
    
    print(f'[DEBUG-disabled-stats] PP rank {pp_rank}: Processing {total_tokens} valid tokens across {num_local_layers} LOCAL layers')
    
    # Step 1: 每个rank统计自己的local layers（使用全局layer编号）
    local_stats = {}
    
    if total_tokens > 0:
        # 获取当前rank的layer offset（全局编号起点）
        from verl.utils.megatron.router_replay_utils import get_current_rank_layer_info
        layer_info = get_current_rank_layer_info(self.tf_config, vp_rank=None)
        layer_offset = layer_info["start"]  # 全局layer起始索引
        
        print(f'[DEBUG-disabled-stats] PP rank {pp_rank}: Layer offset={layer_offset}, processing layers {layer_offset} to {layer_offset + num_local_layers - 1}')
        
        # 处理每一层
        for local_layer_idx in range(num_local_layers):
            global_layer_idx = layer_offset + local_layer_idx  # 全局layer索引
            layer_routing = disabled_layers_top_k_idx[:, local_layer_idx, :]  # (total_tokens, top_k)
            
            valid_indices = layer_routing.flatten()
            
            # Filter negative indices (padding)
            if (valid_indices < 0).any():
                valid_indices = valid_indices[valid_indices >= 0]
            
            if valid_indices.numel() == 0:
                continue
            
            # ✨ 直接使用 layer_name 格式
            layer_name = f"layer_{global_layer_idx}"
            
            if layer_name not in local_stats:
                local_stats[layer_name] = {}
            
            # Count expert usage
            unique_experts = valid_indices.unique()
            for expert_id in unique_experts:
                count = (valid_indices == expert_id).sum().item()
                expert_id_str = str(expert_id.item())
                if expert_id_str not in local_stats[layer_name]:
                    local_stats[layer_name][expert_id_str] = 0
                local_stats[layer_name][expert_id_str] += count
        
        print(f'[DEBUG-disabled-stats] PP rank {pp_rank}: Collected stats for {len(local_stats)} LOCAL layers')
    else:
        print(f'[WARN-disabled] PP rank {pp_rank}: total_tokens=0, no stats to collect')
    
    # Step 2: 使用all_gather_object收集所有PP ranks的统计结果
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    
    print(f'[DEBUG-disabled-gather] PP rank {pp_rank}: About to all_gather_object, pp_world_size={pp_world_size}')
    
    all_ranks_stats = [None] * pp_world_size
    torch.distributed.all_gather_object(
        all_ranks_stats,
        local_stats,
        group=pp_group
    )
    
    print(f'[DEBUG-disabled-gather] PP rank {pp_rank}: all_gather_object SUCCESS! Received {len(all_ranks_stats)} rank results')
    
    # Step 3: 合并所有PP ranks的统计结果
    merged_stats = {}
    for rank_idx, rank_stats in enumerate(all_ranks_stats):
        print(f'[DEBUG-disabled-gather] PP rank {pp_rank}: Merging stats from PP rank {rank_idx}, has {len(rank_stats)} layers')
        for layer_name, expert_counts in rank_stats.items():
            if layer_name not in merged_stats:
                merged_stats[layer_name] = {}
            for expert_id_str, count in expert_counts.items():
                if expert_id_str not in merged_stats[layer_name]:
                    merged_stats[layer_name][expert_id_str] = 0
                merged_stats[layer_name][expert_id_str] += count
    
    print(f'[DEBUG-disabled-gather] PP rank {pp_rank}: After merging all PP ranks, total {len(merged_stats)} layers')
    
    # ✨ V3优化：不再立即保存，而是返回统计数据
    # 保存逻辑移到 forward_backward_batch_patch 中，通过累积器处理
    if merged_stats:
        print(f'[DEBUG-disabled-collect] PP rank {pp_rank}: Collected {len(merged_stats)} layers for step {current_step} (will accumulate, not save yet)')
    else:
        print(f'[WARN-disabled] PP rank {pp_rank}: No merged stats collected')
    
    return merged_stats


def _do_save_expert_stats(moe_patch_dir, model_name, stats_dict, step, mode, dp_rank, num_experts=None, top_k=None):
    """实际执行保存的函数（可以异步调用）
    
    Args:
        moe_patch_dir: 输出目录
        model_name: 模型名称
        stats_dict: {layer_name: {expert_id: count}} (如 {"layer_0": {"0": 123}})
        step: training step number
        mode: "disabled", "r2", or "r3"
        dp_rank: data parallel rank
        num_experts: 专家总数，如果为None则从stats_dict推断
        top_k: top-k值，如果为None则使用默认值8
    """
    import os
    import json
    from datetime import datetime
    import fcntl  # ✅ 文件锁（Linux/Unix）
    
    # 准备输出文件名
    os.makedirs(moe_patch_dir, exist_ok=True)
    step_str = str(step) if step is not None else "unknown"
    output_file = os.path.join(moe_patch_dir, f"verl_moe_lb_step_{step_str}_rank_{dp_rank}.jsonl")
    lock_file = output_file + ".lock"  # ✅ 专用锁文件
    
    # ✅ 使用文件锁保护整个读-改-写过程
    lock_fd = None
    try:
        # 获取排他锁
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        print(f'[DEBUG] DP rank {dp_rank} acquired file lock: {lock_file}')
        
        # ✨ 检查文件是否已存在，如果存在则读取并合并
        existing_data = {}
        file_exists = os.path.exists(output_file)
        
        if file_exists:
            print(f'[INFO] DP rank {dp_rank} found existing file, will merge: {output_file}')
            try:
                with open(output_file, 'r') as f:
                    for line in f:
                        layer_data = json.loads(line)
                        layer = layer_data['layer']
                        existing_data[layer] = layer_data
                print(f'[DEBUG] Loaded {len(existing_data)} layers from existing file')
            except Exception as e:
                print(f'[WARN] Failed to read existing file, will overwrite: {e}')
                existing_data = {}
        else:
            print(f'[INFO] DP rank {dp_rank} creating new file: {output_file}')
        
        # 写入JSONL文件（合并模式）
        timestamp = datetime.now().isoformat()
        
        with open(output_file, 'w') as f:
            # ✨ 简化：stats_dict 的 key 已经是 layer_name 格式了，不需要映射
            # 收集所有需要写入的layers（新数据 + 已存在的数据）
            all_layers = set(stats_dict.keys()) | set(existing_data.keys())
            
            # 按layer顺序写入（按layer编号排序）
            def layer_sort_key(layer_name):
                # layer_name 格式: "layer_0", "layer_1", ...
                return int(layer_name.split('_')[1])
            
            for layer_name in sorted(all_layers, key=layer_sort_key):
                # 获取新数据
                new_expert_counts = stats_dict.get(layer_name, {})
                
                # 获取旧数据（兼容旧格式：可能是字典或列表）
                old_expert_counts = {}
                if layer_name in existing_data:
                    old_assignments = existing_data[layer_name]['actual_assignments']
                    # 处理旧格式：如果是列表，转换为字典；如果是字典，直接使用
                    if isinstance(old_assignments, list):
                        # 新格式：列表，索引对应expert_id
                        for expert_id, count in enumerate(old_assignments):
                            if count > 0:
                                old_expert_counts[str(expert_id)] = count
                    elif isinstance(old_assignments, dict):
                        # 旧格式：字典
                        old_expert_counts = old_assignments
                
                # ✨ 合并：expert counts相加
                merged_expert_counts = {}
                all_expert_ids = set(new_expert_counts.keys()) | set(old_expert_counts.keys())
                
                for expert_id in all_expert_ids:
                    old_count = int(old_expert_counts.get(expert_id, 0))
                    new_count = int(new_expert_counts.get(expert_id, 0))
                    merged_expert_counts[expert_id] = old_count + new_count
                
                # 对merged_expert_counts按expert_id排序（数字顺序；但key是str）
                merged_expert_counts = {k: merged_expert_counts[k] for k in sorted(merged_expert_counts, key=lambda s: int(s))}
                
                # 推断或使用提供的num_experts（优先使用旧数据中的值）
                if layer_name in existing_data:
                    old_data = existing_data[layer_name]
                    if 'num_experts' in old_data and num_experts is None:
                        num_experts = old_data['num_experts']
                    if 'top_k' in old_data and top_k is None:
                        top_k = old_data['top_k']
                
                if num_experts is None:
                    # 从merged_expert_counts推断：最大expert_id + 1
                    if merged_expert_counts:
                        max_expert_id = max(int(k) for k in merged_expert_counts.keys())
                        inferred_num_experts = max_expert_id + 1
                    else:
                        inferred_num_experts = 0
                    layer_num_experts = inferred_num_experts
                else:
                    layer_num_experts = num_experts
                
                # 使用提供的top_k或默认值8
                layer_top_k = top_k if top_k is not None else 8
                
                # 计算总tokens：路由次数 / top_k = 实际token数
                # 在MoE中，每个token在每个layer会被路由到top_k个expert
                # 所以路由次数 = token数 × top_k，因此 token数 = 路由次数 / top_k
                routing_count = sum(merged_expert_counts.values())
                total_tokens = routing_count // layer_top_k if layer_top_k > 0 else 0
                
                # 将actual_assignments从字典转换为列表格式
                # 列表索引对应expert_id，值为count
                actual_assignments_list = [0] * layer_num_experts
                for expert_id_str, count in merged_expert_counts.items():
                    expert_id = int(expert_id_str)
                    if expert_id < layer_num_experts:
                        actual_assignments_list[expert_id] = count
                
                # 构造一行数据
                line_data = {
                    "timestamp": timestamp,
                    "iteration": step if step is not None else -1,
                    "layer": layer_name,
                    "rank": dp_rank,
                    "tokens": total_tokens,
                    "num_experts": layer_num_experts,
                    "top_k": layer_top_k,
                    "actual_assignments": actual_assignments_list
                }
                
                # 写入一行（JSONL格式）
                f.write(json.dumps(line_data) + '\n')
        
        action = "merged and saved" if file_exists else "saved"
        print(f'[INFO] DP rank {dp_rank} {action} {len(all_layers)} layers to {output_file}')
        print(f'[INFO] Mode: {mode}, Step: {step}, DP rank: {dp_rank}, Total layers: {len(all_layers)}')
    
    finally:
        # ✅ 释放文件锁
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
                # 清理锁文件（忽略不存在的错误）
                try:
                    os.remove(lock_file)
                except FileNotFoundError:
                    pass  # 锁文件可能已被其他线程删除，忽略
                print(f'[DEBUG] DP rank {dp_rank} released file lock: {lock_file}')
            except Exception as e:
                print(f'[WARN] Failed to release lock: {e}')


def _merge_stats_into_accumulator(self, stats, step):
    """合并统计数据到 step 累积器（内存操作，不保存文件）
    
    Args:
        stats: {layer_name: {expert_id: count}} 待合并的统计数据
        step: training step number
    
    说明：
        - 将新的统计数据累加到 self._step_accumulated_stats[step]
        - 多个 mini batch 的数据会在内存中合并
        - 只有当 step 变化时才会触发文件保存
    """
    if not hasattr(self, '_step_accumulated_stats'):
        self._step_accumulated_stats = {}
    
    if step not in self._step_accumulated_stats:
        self._step_accumulated_stats[step] = {}
        print(f'[ACCUMULATOR] Initialized accumulator for step {step}')
    
    # 合并 layer 级别的统计
    for layer_name, expert_counts in stats.items():
        if layer_name not in self._step_accumulated_stats[step]:
            self._step_accumulated_stats[step][layer_name] = {}
        
        # 合并 expert 级别的 count
        for expert_id, count in expert_counts.items():
            if expert_id not in self._step_accumulated_stats[step][layer_name]:
                self._step_accumulated_stats[step][layer_name][expert_id] = 0
            self._step_accumulated_stats[step][layer_name][expert_id] += count
    
    print(f'[ACCUMULATOR] Merged {len(stats)} layers into step {step}, total layers: {len(self._step_accumulated_stats[step])}')


def _save_accumulated_stats_for_step(self, step, async_save=True):
    """保存一个 step 的累积统计数据（触发文件写入）
    
    Args:
        step: training step number
        async_save: 是否使用异步保存（默认True）
    
    说明：
        - 将内存中累积的一个完整 step 的数据保存到文件
        - 保存后清理该 step 的内存数据
        - 每个 step 只保存一次，大幅减少文件 I/O
    """
    if not hasattr(self, '_step_accumulated_stats') or step not in self._step_accumulated_stats:
        print(f'[ACCUMULATOR-WARN] No accumulated stats for step {step}, nothing to save')
        return
    
    stats = self._step_accumulated_stats[step]
    
    if not stats:
        print(f'[ACCUMULATOR-WARN] Empty stats for step {step}, skipping save')
        del self._step_accumulated_stats[step]
        return
    
    # 确定模式
    mode = "r2" if getattr(self, "enable_routing_replay", False) else "disabled"
    
    print(f'[ACCUMULATOR-SAVE] Saving accumulated stats for step {step}, mode={mode}, layers={len(stats)}')
    
    # 调用统一的保存函数
    self._save_expert_stats_as_jsonl(
        stats,
        step=step,
        mode=mode,
        async_save=async_save
    )
    
    # 清理已保存的数据
    del self._step_accumulated_stats[step]
    print(f'[ACCUMULATOR-SAVE] Step {step} saved and cleared from memory')


def _save_expert_stats_as_jsonl(self, stats_dict, step=None, mode="unknown", async_save=True):
    """保存expert统计为JSONL格式（统一函数for disabled/R2/R3）
    
    Args:
        stats_dict: {layer_name: {expert_id: count}} 格式的字典
                    例如: {"layer_0": {"0": 123, "1": 456}, "layer_1": {"0": 789}}
                    - layer_name: 层名称，格式为 "layer_{layer_idx}"
                    - expert_id: 专家ID（字符串格式），如 "0", "1", "2"
                    - count: 该专家被路由的次数（路由次数，不是token数）
        step: training step number
        mode: "disabled", "r2", or "r3"
        async_save: 是否使用异步保存（默认True）
    
    输出格式：每个step一个jsonl文件，每行一个layer的数据
    字段：timestamp, iteration, layer, rank, tokens, num_experts, top_k, actual_assignments
    """
    import torch
    from megatron.core import parallel_state as mpu
    
    # ✨ 每个DP rank保存自己的数据
    # 只需要满足：正确的PP rank + TP rank 0
    should_save = False
    if mode == "disabled":
        # Disabled模式：PP rank 0, 每个DP rank, TP rank 0
        should_save = (mpu.get_pipeline_model_parallel_rank() == 0 and 
                      mpu.get_tensor_model_parallel_rank() == 0)
    else:  # R2/R3 mode
        # R2/R3模式：最后一个PP rank, 每个DP rank, TP rank 0
        should_save = (mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1 and
                      mpu.get_tensor_model_parallel_rank() == 0)
    
    if not should_save:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        print(f'[DEBUG-save_jsonl] Rank PP={pp_rank} DP={dp_rank} TP={tp_rank} not responsible for saving (mode={mode}), skipping')
        return
    
    if not stats_dict:
        print(f'[DEBUG-save_jsonl] No stats to save')
        return
    
    # 准备参数
    moe_patch_dir = getattr(self, "_moe_patch_dir", None)
    if not moe_patch_dir:
        print(f'[ERROR-save_jsonl] moe_patch_dir not found')
        return
    
    model_name = getattr(self, "model_name", "model")
    dp_rank = mpu.get_data_parallel_rank()
    
    # 获取num_experts和top_k配置
    num_experts = None
    top_k = None
    
    # 尝试从tf_config获取配置
    if hasattr(self, 'tf_config') and self.tf_config is not None:
        # 尝试从tf_config获取num_experts
        if hasattr(self.tf_config, 'num_experts'):
            num_experts = self.tf_config.num_experts
        elif hasattr(self.tf_config, 'model_config') and hasattr(self.tf_config.model_config, 'num_experts'):
            num_experts = self.tf_config.model_config.num_experts
        
        # 尝试从tf_config获取top_k（num_experts_per_tok）
        if hasattr(self.tf_config, 'num_experts_per_tok'):
            top_k = self.tf_config.num_experts_per_tok
        elif hasattr(self.tf_config, 'top_k'):
            top_k = self.tf_config.top_k
        elif hasattr(self.tf_config, 'model_config') and hasattr(self.tf_config.model_config, 'num_experts_per_tok'):
            top_k = self.tf_config.model_config.num_experts_per_tok
    
    # 如果仍未获取到，尝试从模型配置获取
    if num_experts is None or top_k is None:
        try:
            from verl.utils.megatron_utils import get_model_config
            model_config = get_model_config(self)
            if num_experts is None and hasattr(model_config, 'num_experts'):
                num_experts = model_config.num_experts
            if top_k is None and hasattr(model_config, 'num_experts_per_tok'):
                top_k = model_config.num_experts_per_tok
        except Exception as e:
            print(f'[DEBUG-save_jsonl] Failed to get model config: {e}, will infer from data')
    
    # 决定是同步还是异步保存
    use_async = async_save and hasattr(self, '_async_save_manager') and self._async_save_manager is not None
    
    if use_async:
        # ✨ 异步保存
        print(f'[DEBUG-save_jsonl] Submitting async save task for mode={mode}, step={step}, dp_rank={dp_rank}, num_experts={num_experts}, top_k={top_k}')
        self._async_save_manager.submit(
            _do_save_expert_stats,
            moe_patch_dir=moe_patch_dir,
            model_name=model_name,
            stats_dict=stats_dict.copy(),  # ✨ 复制一份，避免引用问题
            step=step,
            mode=mode,
            dp_rank=dp_rank,
            num_experts=num_experts,
            top_k=top_k
        )
    else:
        # 同步保存（原有行为）
        print(f'[DEBUG-save_jsonl] Using sync save for mode={mode}, step={step}, dp_rank={dp_rank}, num_experts={num_experts}, top_k={top_k}')
        _do_save_expert_stats(
            moe_patch_dir=moe_patch_dir,
            model_name=model_name,
            stats_dict=stats_dict,
            step=step,
            mode=mode,
            dp_rank=dp_rank,
            num_experts=num_experts,
            top_k=top_k
        )


def _finalize_async_saves(self, timeout=300):
    """等待所有异步保存任务完成（在训练结束时调用）
    
    Args:
        timeout: 最大等待时间（秒），默认 300 秒
    
    Returns:
        bool: 是否所有任务都完成
    
    说明：
        - ✨ V3优化：先保存最后一个 step 的累积数据
        - 然后等待所有异步保存任务完成
    """
    # ✨ V3优化：保存最后一个 step 的累积数据
    if hasattr(self, '_current_training_step') and self._current_training_step is not None:
        print(f'[FINALIZE] Saving final step: {self._current_training_step}')
        if hasattr(self, '_save_accumulated_stats_for_step'):
            self._save_accumulated_stats_for_step(self._current_training_step, async_save=True)
        else:
            print(f'[FINALIZE-WARN] _save_accumulated_stats_for_step not found')
    
    # 检查是否还有未保存的 step
    if hasattr(self, '_step_accumulated_stats') and self._step_accumulated_stats:
        print(f'[FINALIZE-WARN] Found {len(self._step_accumulated_stats)} unsaved steps: {list(self._step_accumulated_stats.keys())}')
        for step in list(self._step_accumulated_stats.keys()):
            print(f'[FINALIZE] Saving remaining step: {step}')
            if hasattr(self, '_save_accumulated_stats_for_step'):
                self._save_accumulated_stats_for_step(step, async_save=True)
    
    # 等待所有异步保存任务完成
    if hasattr(self, '_async_save_manager') and self._async_save_manager is not None:
        print(f'[FINALIZE] Waiting for all async save tasks to complete...')
        success = self._async_save_manager.wait_all(timeout=timeout)
        if success:
            print(f'[FINALIZE] All async save tasks completed successfully')
        else:
            print(f'[FINALIZE-WARN] Some async save tasks did not complete within {timeout}s')
        return success
    else:
        print(f'[FINALIZE] No async save manager, nothing to wait')
        return True


def compute_off_old_policy_metrics(
            log_prob, 
            old_log_prob, 
            rollout_log_prob,
            response_mask):
    """计算training policy和old policy的off-policy metrics"""
    import torch
    from verl.utils import torch_functional as verl_F
    SAFETY_BOUND = 20.0

    assert response_mask.any(), "Expected at least one valid token in response_mask"
    print(f"[debug] verl_patched/routed_expert_capture: {response_mask.shape=}")
    metrics = {}

    mean_log_prob_training = verl_F.masked_mean(log_prob, response_mask, axis=-1)  # (batch_size,)
    training_ppl = torch.exp(-mean_log_prob_training).mean()  # Batch mean of per-sequence PPL
    metrics["actor/training_ppl"] = training_ppl.detach().item()

    metrics["actor/training_logp"] = verl_F.masked_mean(log_prob, response_mask).detach().item()  # (1)

    # 2c. old policy perplexity
    mean_log_prob_old = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)  # (batch_size,)
    old_ppl = torch.exp(-mean_log_prob_old).mean()  # Batch mean of per-sequence PPL
    metrics["actor/old_ppl"] = old_ppl.detach().item()

    metrics["actor/old_logp"] = verl_F.masked_mean(old_log_prob, response_mask).detach().item()  # (1)
    
    # 1. 计算log ratio： training policy/old policy
    log_ratio = log_prob - old_log_prob  # (batch_size, response_length)
    metrics["actor/training_old_log_ratio"] = verl_F.masked_mean(log_ratio, response_mask).detach().item()  # (1)

    # 2a. IS token mean: E[exp(log_ratio)]
    is_tm = torch.exp(verl_F.masked_mean(log_ratio, response_mask))
    metrics["actor/IS_token_mean"] = is_tm.detach().item()

    log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    # 2g. IS_safe_token_mean: Importance Sampling token mean using log_ratio_safe
    is_safe_tm = torch.exp(verl_F.masked_mean(log_ratio_safe, response_mask))
    metrics["actor/IS_safe_token_mean"] = is_safe_tm.detach().item()
    
    # 2b. IS token std
    is_token = torch.exp(log_ratio)
    masked_is_token = is_token[response_mask]
    metrics["actor/IS_token_std"] = masked_is_token.std().detach().item()

    # 计算 rollout prob metric
    if rollout_log_prob is not None:

        mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)  # (batch_size,)
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()  # Batch mean of per-sequence PPL 
        metrics["actor/rollout_ppl"] = rollout_ppl.detach().item()
        metrics["actor/rollout_logp"] = verl_F.masked_mean(rollout_log_prob, response_mask).detach().item()  # (1)
        # 计算log ratio： old policy/rollout policy，看的是训推差异
        rollout_log_ratio = old_log_prob - rollout_log_prob
        metrics["actor/old_rollout_log_ratio"] = verl_F.masked_mean(rollout_log_ratio, response_mask).detach().item()  # (1)

    print(f"[debug] verl_patched_v3/compute_routed_expert_capturer_v3 918: {metrics.keys=}")   
    return metrics


def forward_backward_batch_patch(
    self,
    data,  # DataProto - 延迟类型检查
    forward_only=False,
    post_process_fn=None,
    calculate_entropy=False,
    use_dynamic_bsz=False,
    micro_batch_size=None,
    max_token_len=None,
    mini_batch_size=None,):
    """Patched forward_backward_batch with expert statistics collection"""
    import torch
    
    print(f"[DEBUG] forward_backward_batch patched called, enable_routing_replay={self.enable_routing_replay}")
    print(f"[debug] forward_backward_batch patched called, data.meta_info.keys: {data.meta_info.keys()}")
    print(f"[debug] forward_backward_batch patched called, data.batch.keys: {data.batch.keys()}")
    # 提取step信息
    current_step = None
    if hasattr(data, 'meta_info') and isinstance(data.meta_info, dict):
        if 'global_steps' in data.meta_info:
            current_step = data.meta_info['global_steps']
            print(f'[DEBUG] Extracted step from meta_info: {current_step}')

    # 更新当前step
    if current_step is not None:
        self._current_training_step = current_step
        print(f'[DEBUG] Current step: {current_step}')
    
    # 初始化buffers来保存response_mask和routed_experts
    if not hasattr(self, '_disabled_response_mask_buffer'):
        self._disabled_response_mask_buffer = []
    if not hasattr(self, '_disabled_routed_experts_buffer'):
        self._disabled_routed_experts_buffer = []
    
    from megatron.core import parallel_state as mpu
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
    from verl.utils.py_functional import append_to_dict
    from verl.utils.megatron.pipeline_parallel import make_batch_generator
    from verl.utils.torch_functional import broadcast_dict_tensor
    from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
    from verl.utils.megatron_utils import get_model_config, unwrap_model
    from verl.utils.device import get_device_id, get_torch_device
    from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction
    from verl.utils.megatron.router_replay_utils import (
        RouterReplayHelper,
        merge_router_topk_indices,
        pp_gather,
        reorder_and_merge_vpp_layers,
        set_router_replay_data,
    )
    # 数据准备和内存同步
    data.to(get_device_id())
    data.batch = data.batch.contiguous()
    mini_batch = data
    broadcast_dict_tensor(
        mini_batch.batch,
        src=mpu.get_pipeline_model_parallel_last_rank(),
        group=mpu.get_pipeline_model_parallel_group(),
    )
    
    mini_batch.to("cpu")
    mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
    self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
    if self.has_multi_modal_inputs:
        mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
        mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
            list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
        ).to(torch.int64)
    if mini_batch.batch["position_ids"].dim() == 3:
        mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][:, 0]

    indices = None
    temperature = data.meta_info["temperature"]
    if use_dynamic_bsz:
        assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
            micro_batches, indices = rearrange_micro_batches(
                batch=mini_batch.batch,
                num_batches_divided_by=microbatch_group_size_per_vp_stage,
                max_token_len=max_token_len,
            )
            assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, \
                f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage " \
                f"{microbatch_group_size_per_vp_stage} for megatron backend"
        else:
            micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
        total_seqlen = max_token_len
    else:
        assert micro_batch_size is not None, \
            "micro_batch_size is needed to be passed in when not using dynamic batch size"
        micro_batches = mini_batch.batch.split(micro_batch_size)
        seq_len = micro_batches[0]["input_ids"].shape[1]
        total_seqlen = micro_batch_size * seq_len

    n_micro_batch = len(micro_batches)
    forward_backward_func = get_forward_backward_func()

    def loss_func(output, data, meta_info):
        log_probs = None
        entropy = None
        if isinstance(output, dict):
            log_probs = output["log_probs"]
            if "entropy" in output:
                entropy = output["entropy"]
        else:
            assert isinstance(output, torch.Tensor)
            log_probs = output
        device = log_probs.device
        metrics = {}
        if forward_only:
            if post_process_fn is not None:
                stats = post_process_fn(output, data)
                metrics.update(stats)
            if not calculate_entropy:
                return torch.tensor(1.0, device=device), metrics

        responses = data["responses"]
        response_length = responses.size(1)
        response_mask = data["response_mask"].to(bool)
        print(f'[DEBUG-loss_func] response_mask shape: {response_mask.shape}')
        loss_agg_mode = self.config.loss_agg_mode
        log_prob = log_probs[:, -response_length - 1 : -1].contiguous()
        ret_entropy = None
        stats = {}
        if not forward_only:
            old_log_prob = data["old_log_probs"]
            advantages = data["advantages"]
            entropy_coeff = self.config.entropy_coeff
            loss_agg_mode = self.config.loss_agg_mode
            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
            policy_loss_fn = get_policy_loss_fn(loss_mode)
            rollout_is_weights = data.get("rollout_is_weights", None)
            pg_loss, pg_metrics = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                config=self.config,
                rollout_is_weights=rollout_is_weights,
            )
            stats.update(pg_metrics)
            # 计算 training policy 和 old policy 的off metric
            print(f"[debug] verl_patched/routed_expert_capturer 260: loss_func off_policy_metric")
            rollout_log_prob = data.get("rollout_log_probs", None)
            off_policy_metric = compute_off_old_policy_metrics(log_prob, old_log_prob, rollout_log_prob, response_mask)
            stats.update(off_policy_metric)
            
            # 统一移到compute_off_old_policy_metrics 函数
            # rollout_log_prob = data.get("rollout_log_probs", None)
            # print(f"[debug] verl_patched/routed_expert_capturer 1073: {rollout_log_prob=}{loss_mode=}")
            # if loss_mode != "rollout_correction" and rollout_log_prob is not None:
            #     print(f"[debug] verl_patched/routed_expert_capturer 1075: rollout_corr_metrics")
            #     from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
            #     rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
            #         log_prob=log_prob,
            #         rollout_log_prob=rollout_log_prob,
            #         response_mask=response_mask,
            #     )
            #     stats.update(rollout_corr_metrics)
            stats["actor/pg_loss"] = pg_loss.detach().item()
            policy_loss = pg_loss
        if calculate_entropy:
            entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
            if not forward_only:
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                entropy_coeff = meta_info["entropy_coeff"] if meta_info else self.config.entropy_coeff
                policy_loss = pg_loss - entropy_coeff * entropy_loss
            else:
                ret_entropy = entropy
        print(f"[debug] verl_patched/loss_func 980: {self.config.use_kl_loss=}")
        if forward_only:
            policy_loss = torch.tensor(1.0, device=device)
        else:
            if self.config.use_kl_loss:
                ref_log_prob = data["ref_log_prob"]
                print(f"[debug] verl_patched/loss_func 985: {ref_log_prob.shape=}")
                kld = kl_penalty(
                    logprob=log_prob,
                    ref_logprob=ref_log_prob,
                    kl_penalty=self.config.kl_loss_type
                )
                kl_loss = agg_loss(
                    loss_mat=kld,
                    loss_mask=response_mask,
                    loss_agg_mode=self.config.loss_agg_mode,
                )
                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                metrics["actor/kl_loss"] = kl_loss.detach().item()
                metrics["actor/kl_coef"] = self.config.kl_loss_coef
                print(f"[debug] verl_patched/loss_func 1000: {metrics.keys()=}")
            
            # =====R2/R3模式：在loss_func中收集expert统计到self._cumulative_expert_stats
            if getattr(self, "_moe_patch_dir", None) and self.enable_routing_replay:
                self._collect_and_accumulate_r2_stats(data, response_mask)
                
        append_to_dict(metrics, stats)
        return policy_loss, [metrics, ret_entropy]

    def forward_step(batch_iter, model, return_schedule_plan: bool = False):
        if return_schedule_plan:
            assert self.tf_config.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            assert not calculate_entropy, "calculate_entropy must be disabled to return the schedule plan"
            from megatron.core.models.gpt.gpt_model import GPTModel
            assert isinstance(model, GPTModel), "model must be a GPTModel"
            assert self.use_fused_kernels, "use_fused_kernels must be enabled to return the schedule plan"
            from verl.models.mcore.model_forward_1f1b_overlap import gptmodel_forward_1f1b_overlap

        batch = next(batch_iter)
        batch = batch.to(get_device_id())
        batch = batch.contiguous()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(bool)
        position_ids = batch["position_ids"]
        unwrapped_model = unwrap_model(model)
        vp_rank = getattr(unwrapped_model, "vp_stage", 0)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch:
            from verl.utils.model import extract_multi_modal_inputs
            indices = batch.get("multi_modal_inputs_idx", None)
            multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)
        responses = batch["responses"]
        response_length = responses.size(1)
        label = position_ids.clone()
        label[:, -response_length - 1 : -1] = responses
        label_mask = attention_mask.clone()
        label_mask[:, : -response_length - 1] = False
        label_mask[:, -1] = False
        
        # ✨✨✨ 在forward_step中收集response_mask（所有PP ranks都执行）✨✨✨
        if not forward_only and getattr(self, "_moe_patch_dir", None):
            if not getattr(self, "enable_routing_replay", False):  # disabled模式
                from megatron.core import parallel_state as mpu
                
                if not hasattr(self, '_disabled_response_mask_buffer'):
                    self._disabled_response_mask_buffer = []
                
                # 从batch中获取response_mask（所有ranks都能访问）
                response_mask = batch["response_mask"].to(bool)
                # 移到CPU，匹配routed_data的设备
                self._disabled_response_mask_buffer.append(response_mask.clone().cpu())
                
                pp_rank = mpu.get_pipeline_model_parallel_rank()
                print(f'[DEBUG-forward_step] PP rank {pp_rank} saved response_mask, shape: {response_mask.shape}, buffer length: {len(self._disabled_response_mask_buffer)}')

        # ==== disabled mode: RECORD模式用于MoE专家采集
        if not forward_only and getattr(self, "_moe_patch_dir", None) and not getattr(self, "enable_routing_replay", False):
            from megatron.core import parallel_state as mpu
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            
            # ✨ 添加调试信息
            print(f'[DEBUG-RECORD] PP rank {pp_rank}: Checking RouterReplay.router_instances')
            print(f'[DEBUG-RECORD] PP rank {pp_rank}: RouterReplay.router_instances exists: {hasattr(RouterReplay, "router_instances")}')
            if hasattr(RouterReplay, "router_instances"):
                print(f'[DEBUG-RECORD] PP rank {pp_rank}: RouterReplay.router_instances is None: {RouterReplay.router_instances is None}')
                if RouterReplay.router_instances is not None:
                    print(f'[DEBUG-RECORD] PP rank {pp_rank}: RouterReplay.router_instances length: {len(RouterReplay.router_instances)}')
            
            try:
                if RouterReplay.router_instances:
                    router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                    print(f'[DEBUG-RECORD] PP rank {pp_rank}: router_instance_list length: {len(router_instance_list) if router_instance_list else 0}')
                    for router in router_instance_list:
                        router.set_router_replay_action(RouterReplayAction.RECORD)
                    print(f'[DEBUG-RECORD] PP rank {pp_rank}: RECORD mode set successfully')
                else:
                    print(f'[WARN-RECORD] PP rank {pp_rank}: RouterReplay.router_instances is empty! Cannot collect routing data!')
            except Exception as e:
                print(f'[ERROR-RECORD] PP rank {pp_rank}: Failed to set RECORD mode: {e}')

        # ✨ 调试：检查点1
        if not forward_only and getattr(self, "_moe_patch_dir", None) and not getattr(self, "enable_routing_replay", False):
            from megatron.core import parallel_state as mpu
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            print(f'[DEBUG-checkpoint1] PP rank {pp_rank}: After RECORD mode, before forward pass')
        
        # 启动 REPLAY_FORWARD 模式 R2/R3
        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            layers_top_k_idx = batch["routed_experts"]
            set_router_replay_data(layers_top_k_idx, attention_mask, self.tf_config, vp_rank)

        from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn
        if self.use_fused_kernels:
            forward_fn = get_mcore_forward_fused_fn(self.hf_config)
            if return_schedule_plan:
                forward_fn = gptmodel_forward_1f1b_overlap
            output = forward_fn(
                model=model,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=label,
                labels_mask=label_mask,
                temperature=temperature,
                multi_modal_inputs=multi_modal_inputs,
            )
        else:
            forward_fn = get_mcore_forward_fn(self.hf_config)
            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                logits.div_(temperature)
                ret = {}
                if calculate_entropy:
                    logits_bak = logits.clone()
                    entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                else:
                    logits_bak = logits
                log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                log_probs = log_probs.masked_fill(~label_mask, 0.0)
                ret["log_probs"] = log_probs
                return ret
            logits_processor_args = {"label": label, "label_mask": label_mask}
            output = forward_fn(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                multi_modal_inputs=multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                data_format="thd" if self.config.megatron.use_remove_padding else "bshd",
            )
        if forward_only:
            meta_info = None
        else:
            clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
            meta_info = {
                "clip_ratio": self.config.clip_ratio,
                "entropy_coeff": self.config.entropy_coeff,
                "clip_ratio_c": clip_ratio_c,
            }
        # ✨ 调试：检查点2
        if not forward_only and getattr(self, "_moe_patch_dir", None) and not getattr(self, "enable_routing_replay", False):
            from megatron.core import parallel_state as mpu
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            print(f'[DEBUG-checkpoint2] PP rank {pp_rank}: After forward pass, before merge')
            print(f'[DEBUG-checkpoint2] PP rank {pp_rank}: forward_only={forward_only}, _moe_patch_dir={getattr(self, "_moe_patch_dir", None)}, enable_routing_replay={getattr(self, "enable_routing_replay", False)}')
        
        # R2/R3专用控制
        if getattr(self, "enable_routing_replay", False):
            if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
                merge_router_topk_indices(
                    attention_mask, input_ids, self.mini_layer_top_k_idx_list, self.tf_config, vp_rank
                )
            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        # disabled mode 每个micro存到 _disabled_routed_experts_buffer
        if not forward_only and getattr(self, "_moe_patch_dir", None) and not getattr(self, "enable_routing_replay", False):
            from megatron.core import parallel_state as mpu
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            
            before_len = len(self._disabled_routed_experts_buffer)
            print(f'[DEBUG-merge] PP rank {pp_rank}: Before merge_router_topk_indices, buffer length: {before_len}')
            
            merge_router_topk_indices(
                attention_mask, input_ids, self._disabled_routed_experts_buffer, self.tf_config, vp_rank
            )
                
            after_len = len(self._disabled_routed_experts_buffer)
            print(f'[DEBUG-merge] PP rank {pp_rank}: After merge_router_topk_indices, buffer length: {after_len}')
            
            # 验证数据已正确添加
            if after_len > before_len:
                new_data = self._disabled_routed_experts_buffer[-1]
                print(f'[DEBUG-disabled-buffer] PP rank {pp_rank}: Micro_batch {before_len} added to routed_experts_buffer')
                print(f'[DEBUG-disabled-buffer] PP rank {pp_rank}:   Shape: {new_data.shape} (expected: batch_size={attention_mask.shape[0]}, seq_len={attention_mask.shape[1]}, num_local_layers, top_k=8)')
                print(f'[DEBUG-disabled-buffer] PP rank {pp_rank}:   Buffer total length: {len(self._disabled_routed_experts_buffer)}')
            else:
                print(f'[WARN-merge] PP rank {pp_rank}: merge_router_topk_indices did NOT add data! before={before_len}, after={after_len}')
                print(f'[WARN-merge] PP rank {pp_rank}: This means RouterReplay.router_instances might be empty or merge failed')
            
            # 保存之后，清空router_instances
            try:
                if RouterReplay.router_instances:
                    router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                    for router in router_instance_list:
                        router.clear_router_replay_action()
                        router.clear_indices()
                    print(f'[DEBUG-merge] PP rank {pp_rank}: Cleared router instances')
                else:
                    print(f'[WARN-merge] PP rank {pp_rank}: RouterReplay.router_instances is empty, nothing to clear')
            except Exception as e:
                print(f'[ERROR-merge] PP rank {pp_rank}: Failed to clear router instances: {e}')
        return output, partial(loss_func, data=batch, meta_info=meta_info)

    batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,
            micro_batch_size=1,
            forward_only=forward_only,
        )
    else:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,
            micro_batch_size=1,
            forward_only=forward_only,
        )
    if self.has_multi_modal_inputs:
        data.batch.pop("multi_modal_inputs")
        data.batch.pop("multi_modal_inputs_idx")
        data.non_tensor_batch.pop("multi_modal_inputs")
    losses_reduced = {"output": losses_reduced}
    if use_dynamic_bsz:
        losses_reduced["indices"] = indices
    # R2 mode 汇总 mini_layer_top_k_idx_list，在计算old_prob时, 返回 losses_reduced 携带 mini_layer_top_k_idx_tensor
    if getattr(self, "enable_routing_replay", False) and RouterReplayHelper.is_r2_record_action(self.tf_config):
        if getattr(self.tf_config, "virtual_pipeline_model_parallel_size", None) is not None:
            vp_size = len(self.actor_module)
            microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
            bs = n_micro_batch
            losses_reduced["mini_layer_top_k_idx_tensor"] = reorder_and_merge_vpp_layers(
                self.mini_layer_top_k_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
            )
        else:
            losses_reduced["mini_layer_top_k_idx_tensor"] = torch.cat(self.mini_layer_top_k_idx_list, dim=0)
        self.mini_layer_top_k_idx_list = []
        
    # ====R2/R3模式：累积到step累积器（不再立即保存）
    if self._moe_patch_dir and hasattr(self, '_cumulative_expert_stats') and self.enable_routing_replay:
        if self._cumulative_expert_stats:
            print(f'[DEBUG-r2-accumulate] Accumulating R2 stats for step {current_step} (will save when step changes)')
            # ✨ V3优化：累积到内存，不立即保存
            self._merge_stats_into_accumulator(self._cumulative_expert_stats, current_step)
            # 清空临时统计，为下一个mini_batch准备
            self._cumulative_expert_stats = {}

    # ====Disabled mode: 处理expert统计（应用response_mask过滤、pp_gather、收集），然后累积
    if not getattr(self, "enable_routing_replay", False) and self._moe_patch_dir:
        # ✨ V3优化：_collect_disabled_mode_expert_stats 现在返回统计数据而不是直接保存
        disabled_stats = self._collect_disabled_mode_expert_stats(current_step, n_micro_batch)
        
        # 累积到step累积器（不立即保存）
        if disabled_stats:
            print(f'[DEBUG-disabled-accumulate] Accumulating Disabled stats for step {current_step} (will save when step changes)')
            self._merge_stats_into_accumulator(disabled_stats, current_step)
        
        # 清空disabled 模式下 的 buffers
        self._disabled_routed_experts_buffer = []
        self._disabled_response_mask_buffer = []
        RouterReplay.clear_global_indices()
        RouterReplay.clear_global_router_replay_action()
    
        
    
    return losses_reduced


def apply_megatron_ppo_actor_patch():
    from verl.workers.actor.megatron_actor import MegatronPPOActor
    """替换 MegatronPPOActor 的 forward_backward_batch 方法"""
    print("[PATCH] Applying MegatronPPOActor patch...")
    
    # 1. 注入工具方法到类
    # R2模式：在loss_func中收集
    MegatronPPOActor._collect_expert_stats_r2_mode = _collect_expert_stats_r2_mode
    MegatronPPOActor._collect_and_accumulate_r2_stats = _collect_and_accumulate_r2_stats
    MegatronPPOActor._collect_disabled_mode_expert_stats = _collect_disabled_mode_expert_stats
    # V3版优化：Step级别累积方法
    MegatronPPOActor._merge_stats_into_accumulator = _merge_stats_into_accumulator
    MegatronPPOActor._save_accumulated_stats_for_step = _save_accumulated_stats_for_step
    # 统一的保存函数
    MegatronPPOActor._save_expert_stats_as_jsonl = _save_expert_stats_as_jsonl
    # 异步保存完成函数
    MegatronPPOActor._finalize_async_saves = _finalize_async_saves

    
    # 2. 替换 forward_backward_batch 方法（类级别）
    MegatronPPOActor.forward_backward_batch = forward_backward_batch_patch
    print(f"[PATCH] MegatronPPOActor.forward_backward_batch replaced at class level")
    
    # 4. Patch __init__ 以在实例创建时强制绑定 patched 方法
    orig_init = MegatronPPOActor.__init__
    
    @wraps(orig_init)
    def patched_init(self, *args, **kwargs):
        # 执行原始初始化
        orig_init(self, *args, **kwargs)
        print(f"[PATCH] MegatronPPOActor.__init__ executed")
        
        self.model_name = "model"
        if hasattr(self, 'hf_config') and hasattr(self.hf_config, 'name_or_path'):
            self.model_name = os.path.basename(self.hf_config.name_or_path)
        elif hasattr(self, 'config') and hasattr(self.config, 'model') and hasattr(self.config.model, 'path'):
            self.model_name = os.path.basename(self.config.model.path)
        
        self._moe_patch_dir = os.getenv("MOE_PATCH_DIR")
        if self._moe_patch_dir:
            os.makedirs(self._moe_patch_dir, exist_ok=True)
            
            # ✨ 初始化异步保存管理器
            self._async_save_manager = AsyncSaveManager(max_workers=2)
            print(f'[PATCH] AsyncSaveManager initialized (max_workers=2)')

            # ✨ V3优化：初始化 step 级别累积器
            self._step_accumulated_stats = {}  # {step: {layer_name: {expert_id: count}}}
            self._current_training_step = None  # 当前正在处理的 step
            print(f'[PATCH] Step-level accumulator initialized (V3 optimization)')

            if self.enable_routing_replay:
                # R2/R3模式：临时存储当前mini_batch的统计（每次mini_batch后清空）
                print(f'[PATCH] Initializing cumulative expert stats for R2/R3 mode')
                self._cumulative_expert_stats = {}
            else:
                # Disabled模式：临时buffer
                print(f'[PATCH] Initializing disabled expert stats for Disabled mode')
                self._disabled_response_mask_buffer = []
                self._disabled_routed_experts_buffer = []
        else:
            self._async_save_manager = None
            self._step_accumulated_stats = {}
            self._current_training_step = None
 
    MegatronPPOActor.__init__ = patched_init
    print(f"[PATCH] MegatronPPOActor.__init__ patched, id = {id(MegatronPPOActor.__init__)}")

# ================= 3. Patch ActorRolloutRefWorker：添加 expert stats 保存接口 =================

def apply_actor_rollout_ref_worker_patch():
    from omegaconf import DictConfig, OmegaConf
    from verl.workers.megatron_workers import ActorRolloutRefWorker
    from verl.utils.megatron.router_replay_patch import apply_router_replay_patch
    from verl.single_controller.base.decorator import Dispatch, register
    
    """Patch ActorRolloutRefWorker：在 Worker 初始化时应用 router replay patch，并添加 expert stats 保存接口"""
    
    # ========== 3.1 Patch __init__ ==========
    orig_init = ActorRolloutRefWorker.__init__
    print(f"[PATCH] actor_rollout_ref_worker orig_init id: {id(orig_init)}")
    @wraps(orig_init)
    def patched_init(self, *args, **kwargs):
        # 在worker进程中显式确保patch已应用（针对Ray worker）
        # 这一步很关键：Ray worker是独立进程，需要确保模块级别的patch代码已执行
        moe_patch_dir = os.getenv("MOE_PATCH_DIR")
        if moe_patch_dir:
            print(f"[PATCH-Worker] Worker initializing in process {os.getpid()}, MOE_PATCH_DIR={moe_patch_dir}")
            # 确保patch已在当前worker进程中应用
            # 由于模块已导入，这里调用apply()会检查进程ID，只在新进程中执行
            apply()
        
        orig_init(self, *args, **kwargs)
        
        # 如果设置了 MOE_PATCH_DIR，应用 router replay patch
        if self._is_actor and moe_patch_dir:
            print(f"[PATCH] Worker __init__: Applying router replay patch")
            apply_router_replay_patch()
            # 注意一定要在 apply 之后，此时的 TransformerConfig 才有 enable_routing_replay 这个字段，否则报错unexpected keyword argument
            # Note: override_transformer_config is likely a dict, not an object; use dict syntax
                        # 注意一定要在 orig_init和 apply 之后，此时的 TransformerConfig 才有 enable_routing_replay 这个字段，否则报错unexpected keyword argument
            # Note: override_transformer_config is likely a dict, not an object; use dict syntax
            override_transformer_config = OmegaConf.to_container(OmegaConf.create(self.config.actor.megatron.get("override_transformer_config", {})))
            override_transformer_config["enable_routing_replay"] = True
            self.config.actor.megatron.override_transformer_config = override_transformer_config

    
    ActorRolloutRefWorker.__init__ = patched_init

    # ========== 3.2 添加 save_expert_stats_for_step 方法 ==========
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_expert_stats_for_step(self, step):
        """保存指定 step 的 expert 统计数据
        
        Args:
            step: 要保存的 step 编号
            
        Returns:
            bool: True 如果保存成功，False 如果失败或没有数据
            
        Note:
            这个方法由 ray_trainer 在 update_actor 之后调用，
            用于保存当前 step 的 expert 统计数据。
        """
        if self._is_actor and hasattr(self, 'actor') and hasattr(self.actor, '_save_accumulated_stats_for_step'):
            if hasattr(self.actor, '_step_accumulated_stats') and step in self.actor._step_accumulated_stats:
                print(f'[Worker] Saving expert stats for step {step} (after update_actor)...')
                try:
                    self.actor._save_accumulated_stats_for_step(step, async_save=True)
                    print(f'[Worker] Expert stats for step {step} saved successfully')
                    return True
                except Exception as e:
                    print(f'[Worker-ERROR] Failed to save expert stats for step {step}: {e}')
                    return False
            else:
                print(f'[Worker-DEBUG] No accumulated stats found for step {step}, skipping save')
                return False
        else:
            # Not an actor or actor doesn't have the method, nothing to save
            return False
    
    ActorRolloutRefWorker.save_expert_stats_for_step = save_expert_stats_for_step
    print(f"[PATCH] ActorRolloutRefWorker.save_expert_stats_for_step added")
    
    # ========== 3.3 添加 finalize_async_saves 方法 ==========
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finalize_async_saves(self, timeout=300):
        """Finalize expert stats async saves (wait for all tasks to complete)
        
        Args:
            timeout: Maximum wait time in seconds (default 300)
        
        Returns:
            bool: True if all tasks completed successfully
        
        Note:
            This method is called by ray_trainer at the end of training
            to ensure all expert statistics are properly saved.
        """
        if self._is_actor and hasattr(self, 'actor') and hasattr(self.actor, '_finalize_async_saves'):
            print(f'[Worker] Finalizing expert stats async saves (timeout={timeout}s)...')
            try:
                success = self.actor._finalize_async_saves(timeout=timeout)
                if success:
                    print(f'[Worker] Expert stats async saves completed successfully')
                else:
                    print(f'[Worker-WARN] Expert stats async saves did not complete within {timeout}s')
                return success
            except Exception as e:
                print(f'[Worker-ERROR] Failed to finalize async saves: {e}')
                return False
        else:
            # No async save manager or not an actor, nothing to finalize
            return True
    
    ActorRolloutRefWorker.finalize_async_saves = finalize_async_saves
    print(f"[PATCH] ActorRolloutRefWorker.finalize_async_saves added")


# ================= 4. Patch default_compute_score：已迁移到配置文件方式 =================
# 
# 注意：default_compute_score 的自定义实现已迁移到独立文件：
#   /aistudio/workspace/huilian_ssd/moe_experi/code/verl_patched/custom_compute_score.py
# 
# 现在通过配置文件 custom_reward_function 机制加载，不再使用 patch 方式。
# 配置文件位置：verl/trainer/config/ppo_megatron_trainer.yaml
# 
# 如需自定义 reward 计算逻辑，请修改 custom_compute_score.py 文件。


# ================= 保护机制：确保 patch 可以安全地多次执行 =================
# 使用模块级别的标志来防止重复执行（但允许在不同进程中执行）
_patch_applied_flag = False
_patch_applied_process_id = None

def apply():
    """应用所有 patch，带保护机制确保可以安全地多次调用"""
    global _patch_applied_flag, _patch_applied_process_id
    
    import os
    current_process_id = os.getpid()
    
    # 如果已经在当前进程中执行过，跳过
    if _patch_applied_flag and _patch_applied_process_id == current_process_id:
        print(f"[DEBUG-routed_expert_capturer_v3] Patch already applied in process {current_process_id}, skipping...")
        return
    
    print(f"[DEBUG-routed_expert_capturer_v3] Starting patch in process {current_process_id}...")
    apply_megatron_ppo_actor_patch()
    apply_actor_rollout_ref_worker_patch()
    # Note: apply_reward_score_patch() removed - now using custom_reward_function config instead  

    
    _patch_applied_flag = True
    _patch_applied_process_id = current_process_id
    print(f"[DEBUG-routed_expert_capturer_v3] Patch completed in process {current_process_id}.")


# 只有在设置了 MOE_PATCH_DIR 时才执行
if os.getenv("MOE_PATCH_DIR"):
    print(f"[DEBUG-routed_expert_capturer_v3] MOE_PATCH_DIR {os.getenv('MOE_PATCH_DIR')}set, applying patch")
    apply()
else:
    print(f"[DEBUG-routed_expert_capturer_v3] MOE_PATCH_DIR not set, skipping patch")
