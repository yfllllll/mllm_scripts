curl -X POST "http://localhost:9020/detect_file" \
  -F "file=@/mnt/data/lyf/qwen_agent/1.png" \
  -F "query=详细描述图片内容"