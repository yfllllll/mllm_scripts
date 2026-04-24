docker run -itd --name qwenvl \
  --user root \
  --privileged \
  --net=host \
  --shm-size=32g \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /data1:/data1 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -e ASCEND_RT_VISIBLE_DEVICES=1 \
  vllm-ascend:qwen3_5-v0-a2 bash