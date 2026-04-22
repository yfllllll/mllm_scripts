import gradio as gr 
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from PIL import Image  
from rex_omni import RexOmniWrapper  
import numpy as np  
# 初始化模型  
rex_model = RexOmniWrapper(  
    model_path="IDEA-Research/Rex-Omni",  
    backend="transformers"  
)  
  
def describe_image(image, prompt):  
    """直接使用底层模型进行图像描述"""  
    if image is None:  
        return "请先上传图像"  
      
    # 转换为 PIL Image  
    if isinstance(image, np.ndarray):  
        image = Image.fromarray(image)  
      
    # 构建消息 - 直接使用自定义提示词  
    messages = [  
        {"role": "system", "content": "You are a helpful assistant"},  
        {  
            "role": "user",  
            "content": [  
                {"type": "image", "image": image},  
                {"type": "text", "text": prompt}  
            ]  
        }  
    ]  
      
    # 直接调用底层生成方法  
    if rex_model.backend == "transformers":  
        output, _ = rex_model._generate_transformers(messages)  
    else:  
        output, _ = rex_model._generate_vllm(messages)  
      
    return output  
  
# 创建界面  
with gr.Blocks() as demo:  
    gr.Markdown("# Rex-Omni 图像描述测试")  
      
    with gr.Row():  
        with gr.Column():  
            input_image = gr.Image(label="上传图像", type="numpy")  
            prompt = gr.Textbox(  
                label="输入提示词",  
                placeholder="请描述这张图片",  
                value="请详细描述这张图片的内容"  
            )  
            run_button = gr.Button("生成描述", variant="primary")  
          
        with gr.Column():  
            output_text = gr.Textbox(label="模型输出", lines=15)  
      
    run_button.click(  
        fn=describe_image,  
        inputs=[input_image, prompt],  
        outputs=output_text  
    )  
  
demo.launch(server_name="0.0.0.0",server_port=9005,)