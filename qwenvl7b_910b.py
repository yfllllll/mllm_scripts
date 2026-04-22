 # 设置容器名称
export CONTAINER_NAME=qwen2.5-vl-7b-instruct

 # 选择镜像
export IMG_NAME=swr.cn-south-1.myhuaweicloud.com/ascendhub/qwen2.5-vl-7b-instruct:7.1.T2-800I-A2-aarch64
LOCAL_CACHE_PATH=/app/lyf/qwen2.5-vl-7b-instruct
 # 启动推理微服务，使用ASCEND_VISIBLE_DEVICES选择卡号，范围[0，7]，示例选择0,1卡
 docker run -itd \
     --name=$CONTAINER_NAME \
     -e ASCEND_VISIBLE_DEVICES=0,1 \
     -e MIS_CONFIG=atlas800ia2-2x32gb-bf16-vllm-default \
     -v $LOCAL_CACHE_PATH:/opt/mis/.cache \
     -p 8000:8000 \
     --shm-size 1gb \
     $IMG_NAME 
     

from openmind_hub import snapshot_download
snapshot_download(repo_id="MindSDK/Qwen2.5-VL-7B-Instruct", token="ac1e2208af4d77d82bf48e0027586f9e3930a164", repo_type="model",local_dir="./qwen2.5-vl-7b-instruct")     