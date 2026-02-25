# config
MODEL_NAME="Qwen3-30B-A3B-Instruct-2507"
DATASET="aime24"
MODEL_PATH="/aistudio/workspace/huilian_ssd/model/$MODEL_NAME"
DATASET_PATH="/aistudio/workspace/huilian_ssd/moe_experi/benchmark/$DATASET"

# moe_patch
export MOE_PATCH_DIR="/aistudio/workspace/huilian_ssd/moe_patch/examples/outputs"
MOE_PATCH_EVAL_SCOPE_DIR="/aistudio/workspace/huilian_ssd/moe_patch/src/patch/evalscope"
export PYTHONPATH="${MOE_PATCH_EVAL_SCOPE_DIR}:${PYTHONPATH:-}"

echo "=============================="
echo "评测模型: $MODEL_NAME"
echo "评测数据集: $DATASET"
echo "=============================="

# 清理已有的 vLLM 进程，避免多实例通过 SO_REUSEPORT 共享端口导致 500 错误
echo "🧹 清理所有残留 vLLM API server 进程..."
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
fuser -k 8801/tcp 2>/dev/null || true
sleep 3
if fuser 8801/tcp 2>/dev/null | grep -q '[0-9]'; then
  echo "   ⚠️  端口 8801 仍被占用，再次清理..."
  fuser -k 8801/tcp 2>/dev/null || true
  sleep 3
fi
echo "   ✅ 残留进程清理完成"

# 启动 vLLM 服务（日志写入临时文件，用于检测启动完成）
VLLM_LOG="$MOE_PATCH_DIR/vllm_startup.log"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --enforce-eager \
  --port 8801 > >(tee "$VLLM_LOG") 2>&1 &

vllm_pid=$!
echo "vLLM API 服务启动中 (PID: $vllm_pid, 日志: $VLLM_LOG)..."

# 脚本退出时自动清理 vLLM 进程
cleanup() {
  echo "🛑 清理 vLLM 进程 (PID: $vllm_pid)..."
  kill -9 $vllm_pid 2>/dev/null
  wait $vllm_pid 2>/dev/null
}
trap cleanup EXIT

# 等待服务启动
timeout=1800
elapsed=0
while true; do
  sleep 5
  elapsed=$((elapsed + 5))
  if grep -q "Application startup complete" "$VLLM_LOG" 2>/dev/null; then
    echo "✅ vLLM 模型加载完毕，开始执行评测..."
    break
  fi
  if ! kill -0 $vllm_pid 2>/dev/null; then
    echo "❌ 错误: vLLM 进程意外退出，查看日志: $VLLM_LOG"
    exit 1
  fi
  if [ $elapsed -ge $timeout ]; then
    echo "❌ 错误: vLLM 服务启动超时（${timeout}秒），查看日志: $VLLM_LOG"
    kill -9 $vllm_pid 2>/dev/null
    exit 1
  fi
done

# 执行评测
echo "📊 开始评测 $DATASET..."
evalscope eval \
  --model "$MODEL_NAME" \
  --api-url http://127.0.0.1:8801/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets "$DATASET" \
  --dataset-args "{\"$DATASET\": {\"local_path\": \"$DATASET_PATH\"}}" \
  --generation-config '{"max_tokens": 30000, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "n": 1}' \
  --limit 1
