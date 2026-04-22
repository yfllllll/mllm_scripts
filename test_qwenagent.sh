# Start the model server， set cuda device
 CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/lyf/Qwen3-VL-4B-Instruct \
    --served-model-name Qwen3-VL-4B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 40960 \
    --api-key EMPTY

# Start the FastAPI server

uvicorn main:app --host 0.0.0.0 --port 9020 --reload

# Test the API
curl -X POST "http://localhost:9020/detect" \  
  -F "image_base64=$(base64 -w 0 /mnt/data/lyf/qwen_agent/1.png)" \  
  -F "query=检测图片里的汽车" \  
  -F "categories=car,person"