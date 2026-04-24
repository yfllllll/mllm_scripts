CUDA_VISIBLE_DEVICES=1 python -m falcon_perception.server \  
    --config.hf-local-dir /mnt/disk/lyf/Falcon \  
    --config.num-gpus 1 \  
    --config.port 9001
