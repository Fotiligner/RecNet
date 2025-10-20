GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

model=""
model_name=Qwen2.5-32B-Instruct

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${model} \
  --served-model-name ${model_name} --port "$2" \
  --max_model_len 16384 \
  --trust-remote-code --tensor-parallel-size "${GPU_COUNT}" --gpu-memory-utilization 0.95 --swap-space 0 --max-num-seqs 1024 \
  --disable-custom-all-reduce --enforce-eager --seed 42 \
  --enable-prefix-caching \
  --uvicorn-log-level warning --disable-log-requests
