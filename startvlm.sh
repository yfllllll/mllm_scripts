 CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/lyf/Qwen3-VL-4B-Instruct \
    --served-model-name Qwen3-VL-4B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 40960 \
    --api-key EMPTY