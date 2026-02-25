#!/usr/bin/env python3
"""
启动 vLLM 并应用 Expert 统计 patch
"""

import sys
import os

# 设置路径，确保子进程能找到 patch
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
pythonpath = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = f"{script_dir}:{pythonpath}" if pythonpath else script_dir

if __name__ == "__main__":
    # 应用 patch
    try:
        import vllm_patch
        vllm_patch.apply()
    except:
        pass
    
    # 启动 vLLM
    if len(sys.argv) > 2 and sys.argv[1] == '-m':
        module_name = sys.argv[2]
        sys.argv = [module_name] + sys.argv[3:]
        
        import runpy
        runpy.run_module(module_name, run_name="__main__")
    else:
        print("使用方法: python start_vllm_with_stats.py -m vllm.entrypoints.openai.api_server [参数...]")
        sys.exit(1)
