import torch  
import gradio as gr  
from diffusers import Flux2Pipeline  
from diffusers.utils import load_image  
from typing import List, Optional  
import numpy as np  
  
# 配置 - 适合双 A100 GPU  
repo_id = "/mnt/data/lyf/FLUX.2-dev"  
torch_dtype = torch.bfloat16  
  
# 加载模型 - 使用多GPU配置  
print("正在加载 FLUX.2 模型...")  
pipe = Flux2Pipeline.from_pretrained(  
    repo_id,   
    torch_dtype=torch_dtype,  
    device_map="balanced",  # 自动平衡分布到所有GPU  
    max_memory={2: "70GB", 3: "70GB"}  # 使用GPU 0和1，各限制70GB  
)  
# 使用 device_map 时不需要手动调用 .to(device)  
print("模型加载完成！")  
  
def generate_text_to_image(prompt: str, width: int = 1360, height: int = 768, num_inference_steps: int = 50, guidance_scale: float = 4.0, seed: Optional[int] = None):  
    """模式1: 纯文本生成"""  
    if seed is not None:  
        generator = torch.Generator().manual_seed(seed)  
    else:  
        generator = None  
      
    image = pipe(  
        prompt=prompt,  
        width=width,  # 添加宽度参数  
        height=height,  # 添加高度参数  
        num_inference_steps=num_inference_steps,  
        guidance_scale=guidance_scale,  
        generator=generator,  
    ).images[0]  
      
    return image  
  
def generate_image_to_image(prompt: str, input_image, width: int = 1360, height: int = 768, num_inference_steps: int = 50, guidance_scale: float = 4.0, seed: Optional[int] = None):  
    """模式2: 图像编辑"""  
    if seed is not None:  
        generator = torch.Generator().manual_seed(seed)  
    else:  
        generator = None  
    
    # 如果输入是多张图片，取第一张
    if isinstance(input_image, list) and len(input_image) > 0:
        input_image = input_image[0][0]
      
    image = pipe(  
        prompt=prompt,  
        image=[input_image],  
        width=width,  # 添加宽度参数  
        height=height,  # 添加高度参数  
        num_inference_steps=num_inference_steps,  
        guidance_scale=guidance_scale,  
        generator=generator,  
    ).images[0]  
      
    return image  
  
def generate_multi_image_fusion(prompt: str, input_images: List, width: int = 1360, height: int = 768, num_inference_steps: int = 50, guidance_scale: float = 4.0, seed: Optional[int] = None):  
    """模式3: 多图融合"""  
    if not input_images:  
        return None  
      
    if seed is not None:  
        generator = torch.Generator().manual_seed(seed)  
    else:  
        generator = None  
      
    # 过滤空图像  
    image_list = None
    if input_images is not None and len(input_images) > 0:
        image_list = []
        for item in input_images:
            image_list.append(item[0])
      
    image = pipe(  
        prompt=prompt,  
        image=image_list,  
        width=width,  # 添加宽度参数  
        height=height,  # 添加高度参数  
        num_inference_steps=num_inference_steps,  
        guidance_scale=guidance_scale,  
        generator=generator,  
    ).images[0]  
      
    return image  
  
# 创建 Gradio 界面  
with gr.Blocks(title="FLUX.2 多模式测试") as demo:  
    gr.Markdown("# FLUX.2 多模式图像生成测试")  
    gr.Markdown("基于双 A100 GPU 配置")  
      
    with gr.Tabs():  
        # 模式1: 文本生成  
        with gr.TabItem("模式1: 文本生成"):  
            with gr.Row():  
                with gr.Column():  
                    t2i_prompt = gr.Textbox(label="输入提示词", lines=3, placeholder="描述你想要生成的图像...")  
                    t2i_steps = gr.Slider(minimum=28, maximum=100, value=50, step=1, label="推理步数")  
                    t2i_guidance = gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="引导强度") 
                    t2i_width = gr.Slider(minimum=256, maximum=2048, value=1360, step=16, label="图像宽度")  
                    t2i_height = gr.Slider(minimum=256, maximum=2048, value=768, step=16, label="图像高度") 
                    t2i_seed = gr.Number(label="随机种子 (可选)", value=None, precision=0)  
                    t2i_generate = gr.Button("生成图像", variant="primary")  
                  
                with gr.Column():  
                    t2i_output = gr.Image(label="生成的图像", type="pil")  
              
            t2i_generate.click(  
                fn=generate_text_to_image,  
                inputs=[t2i_prompt, t2i_width, t2i_height, t2i_steps, t2i_guidance, t2i_seed],  
                outputs=t2i_output  
            )  
          
        # 模式2: 图像编辑  
        with gr.TabItem("模式2: 图像编辑"):  
            with gr.Row():  
                with gr.Column():  
                    i2i_prompt = gr.Textbox(label="输入提示词", lines=3, placeholder="描述你想要的编辑效果...")  
                    i2i_gallery = gr.Gallery(
                        label="上传图像（选择一张进行编辑）",
                        type="pil",
                        height="auto",
                        columns=3
                    )  # 移除了 .style() 方法调用
                    i2i_steps = gr.Slider(minimum=28, maximum=100, value=50, step=1, label="推理步数")  
                    i2i_guidance = gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="引导强度") 
                    i2i_width = gr.Slider(minimum=256, maximum=2048, value=1360, step=16, label="图像宽度")  
                    i2i_height = gr.Slider(minimum=256, maximum=2048, value=768, step=16, label="图像高度") 
                    i2i_seed = gr.Number(label="随机种子 (可选)", value=None, precision=0)  
                    i2i_generate = gr.Button("编辑图像", variant="primary")  
                  
                with gr.Column():  
                    i2i_output = gr.Image(label="编辑后的图像", type="pil")  
              
            i2i_generate.click(  
                fn=generate_image_to_image,  
                inputs=[i2i_prompt, i2i_gallery,i2i_width, i2i_height, i2i_steps, i2i_guidance, i2i_seed],  
                outputs=i2i_output  
            )  
          
        # 模式3: 多图融合  
        with gr.TabItem("模式3: 多图融合"):  
            with gr.Row():  
                with gr.Column():  
                    multi_prompt = gr.Textbox(label="输入提示词", lines=3, placeholder="描述融合效果...")  
                      
                    # 使用Gallery组件替代多个独立的Image组件
                    multi_gallery = gr.Gallery(
                        label="上传多张图像进行融合（最多10张）",
                        type="pil",
                        height="auto",
                        columns=5
                    )  # 移除了 .style() 方法调用
                      
                    multi_steps = gr.Slider(minimum=28, maximum=100, value=50, step=1, label="推理步数")  
                    multi_guidance = gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="引导强度")  
                    multi_width = gr.Slider(minimum=256, maximum=2048, value=1360, step=16, label="图像宽度")  
                    multi_height = gr.Slider(minimum=256, maximum=2048, value=768, step=16, label="图像高度") 
                    multi_seed = gr.Number(label="随机种子 (可选)", value=None, precision=0)  
                    multi_generate = gr.Button("融合图像", variant="primary")  
                  
                with gr.Column():  
                    multi_output = gr.Image(label="融合后的图像", type="pil")  
              
            multi_generate.click(  
                fn=generate_multi_image_fusion,  
                inputs=[multi_prompt, multi_gallery, multi_width, multi_height,multi_steps, multi_guidance, multi_seed],  
                outputs=multi_output  
            )  
  
if __name__ == "__main__":  
    demo.launch(share=True, server_name="0.0.0.0", server_port=9005)