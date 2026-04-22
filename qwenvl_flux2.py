import os
import time

# ==========================================
# 1. 环境隔离 (必须放在最顶部)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1,6,7"
# 针对 vLLM 的特殊环境设置
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" 

import torch
import gradio as gr
from diffusers import Flux2Pipeline
from PIL import Image

# 延迟导入 vLLM
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info



# 定义全局变量占位符
llm_engine = None
sampling_params = None
pipe = None
processor = None

def prepare_inputs(messages):
        """准备Qwen-VL输入"""
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }


# ==========================================
# 2. 推理逻辑封装
# ==========================================

def analyze_with_vllm(image_pil, categories_str):
    if image_pil is None or not categories_str:
        return "等待输入...", "等待输入..."
    
    # 使用 PID 区分临时文件，防止多进程冲突


    prompt = f"""
作为专业的图像合成师，请分析这张背景图片，从候选类别中选择适合添加到这个场景的所有物体类别
候选类别：{categories_str}，生成一段高质量的英文图像生成提示词，用于生成要插入的物体
   - 描述物体的位置（如：on the grass, on the table, in the sky）
   - 描述物体的大小和数量
   - 描述与场景的互动关系
   - 确保光照、阴影和风格与背景一致
   - 包含构图和视角信息

请输出格式如下：
【生成提示词】：<结果>
"""
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_pil,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
    
    try:
        # 使用全局的 llm_engine
        inputs = [prepare_inputs(messages)]
        outputs = llm_engine.generate(inputs, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # 简单解析
        if "【生成提示词】：" in generated_text:
            # cats = generated_text.split("【生成提示词】：")[0].replace("【合适类别】：", "").strip()
            cats =  generated_text
            final_p = generated_text.split("【生成提示词】：")[1].strip()
        else:
            cats = "解析失败"
            final_p = generated_text
            
        return cats, final_p
    except Exception as e:
        return f"分析出错: {str(e)}", "无法生成"
  

def run_workflow(bg_image, categories, width, height, steps, guidance, seed):
    # 1. 调用 Qwen (GPU 1)
    suitable_cats, flux_prompt = analyze_with_vllm(bg_image, categories)
    
    if "错误" in suitable_cats or not flux_prompt:
        return suitable_cats, flux_prompt, None

    # 2. 调用 FLUX (GPU 6, 7)
    generator = torch.Generator().manual_seed(int(seed)) if seed >= 0 else None
    
    try:
        # 这里需要注意，FLUX 的 pipeline 也要用全局的
        output_img = pipe(
            prompt=flux_prompt,
            image=bg_image,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        return suitable_cats, flux_prompt, output_img
    except Exception as e:
        return suitable_cats, flux_prompt, None

# ==========================================
# 3. 主进程启动逻辑
# ==========================================

if __name__ == "__main__":
    # 强制设置多进程启动方法
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- 初始化 Qwen-VL (物理卡 1) ---
    print(">>> 正在加载 Qwen-VL (vLLM) 于 GPU 1...")
    qwenvl_model_path = "/mnt/data/lyf/Qwen3-VL-4B-Instruct"
    llm_engine = LLM(
        model=qwenvl_model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.5, # 显存占用降至50%，增加稳定性
        tensor_parallel_size=1,     # 单卡
        dtype="bfloat16",
        max_model_len=41960,
    )
    sampling_params = SamplingParams(
        temperature=0.1, 
        max_tokens=1024,
        # stop=["<|endoftext|>", "<|im_end|>"]
    )
    processor = AutoProcessor.from_pretrained(qwenvl_model_path)
    # --- 初始化 FLUX.2 (物理卡 6, 7) ---
    print(">>> 正在加载 FLUX.2 于 GPU 6, 7...")
    # 逻辑 0 映射卡 1 (给vLLM)，逻辑 1,2 映射卡 6,7 (给FLUX)
    flux_max_mem = {0: "0GB", 1: "70GB", 2: "70GB"}
    pipe = Flux2Pipeline.from_pretrained(
        "/mnt/data/lyf/FLUX.2-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=flux_max_mem
    )

    # --- 启动 Gradio ---
    with gr.Blocks(title="Smart Insertion") as demo:
        gr.Markdown("# 🚀 智能物体插入 (Qwen-VL + FLUX)")
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="上传背景", type="pil")
                txt_input = gr.Textbox(label="输入物体类别", placeholder="例如: a cat, a futuristic car")
                with gr.Accordion("生成设置", open=False):
                    w_s = gr.Slider(512, 1536, 1360, step=8, label="宽")
                    h_s = gr.Slider(512, 1024, 768, step=8, label="高")
                    step_s = gr.Slider(20, 50, 30, step=1, label="步数")
                    seed_s = gr.Number(label="种子 (-1为随机)", value=-1)
                btn = gr.Button("开始分析并生成", variant="primary")
            
            with gr.Column():
                out_cats = gr.Textbox(label="Qwen 适宜性分析")
                out_prompt = gr.Textbox(label="Qwen 生成提示词", lines=5)
                out_image = gr.Image(label="最终生成结果")

        btn.click(
            run_workflow,
            inputs=[img_input, txt_input, w_s, h_s, step_s, gr.State(3.5), seed_s],
            outputs=[out_cats, out_prompt, out_image]
        )

    print(">>> 所有模型加载完毕，正在启动 Gradio 服务...")
    demo.queue().launch(server_name="0.0.0.0", server_port=9009)