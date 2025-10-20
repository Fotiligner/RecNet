GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

model=""
model_name=DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${model} \
  --quantization bitsandbytes --load-format bitsandbytes \
  --enable-reasoning --reasoning-parser deepseek_r1 \
  --served-model-name ${model_name} --port "$2" \
  --trust-remote-code --pipeline-parallel-size "${GPU_COUNT}" --gpu-memory-utilization 0.95 --swap-space 0 --max-num-seqs 1024 \
  --disable-custom-all-reduce --enforce-eager --seed 42 \
  --enable-prefix-caching \
  --uvicorn-log-level warning --disable-log-requests \
  --max_model_len 65536