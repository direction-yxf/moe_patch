"""
Python 自动导入此文件，用于在子进程中应用 patch
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if os.environ.get('MOE_PATCH_DIR'):
    try:
        import vllm_patch
    except:
        pass
