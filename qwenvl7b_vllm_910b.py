import os
from PIL import Image
from vllm import LLM, SamplingParams


# 环境变量设置


def run_qwen2_5_vl():
    model_name = "/opt/mis/lyf/Qwen2.5-VL-7B-Instruct"

    llm = LLM(
        model=model_name,
        #tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        #max_model_len=4096,
    )
    return llm

def main():
    # 初始化模型
    print("Loading model...")
    llm = run_qwen2_5_vl()
    print("Model loaded successfully!")
    image_local_path = "/opt/mis/lyf/recoAll_attractions.jpg"
    data = Image.open(image_local_path).convert("RGB")
    prompt = "请描述这张图片的内容。"
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=4096)
    placeholder = "<|image_pad|>"
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{prompt}<|im_end|>\n"
              "<|im_start|>assistant\n")    
    
    inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                'image': data
            },
        }
    outputs = llm.generate([inputs], sampling_params)
    output_text = [output.outputs[0].text for output in outputs]

    # 解析检测框
    print(output_text)
    print("Output:", output_text)

if __name__ == "__main__":
    main()
