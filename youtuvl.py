# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import transformers;
from transformers import AutoProcessor, AutoModelForCausalLM
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import re
from io import BytesIO
import logging
transformers.logging.set_verbosity_info()  # 设置日志级别为 INFO
logging.basicConfig(level=logging.INFO)  # 设置基础日志配置
# 颜色定义
additional_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
                    'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta', 
                    'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 
                    'lavender', 'violet', 'gold', 'silver']

class YoutuVLModel:
    """Youtu-VL 模型类（使用Transformers）"""
    def __init__(self, model_path="/mnt/disk/lyf/Youtu-VL-4B-Instruct"):
        print("正在加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,  # 使用float16加速
            device_map="auto",  # 自动分配设备
            trust_remote_code=True
        ).eval()
        
        print("正在加载processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True
        )
        print("模型加载完成！")
    
    def parse_youtu_vl_response(self, response_text):
        """解析Youtu VL模型的响应格式"""
        result = []
        
        # 匹配格式: <ref>标签</ref><box><x_X><y_Y><x_X><y_Y></box>
        pattern = r'<ref>(.*?)</ref>\s*<box>(.*?)</box>'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for label, box_content in matches:
            # 解析box坐标
            box_pattern = r'<x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)>'
            box_match = re.search(box_pattern, box_content)
            
            if box_match:
                x1, y1, x2, y2 = map(int, box_match.groups())
                # 确保坐标顺序正确
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                    
                result.append({
                    "label": label.strip(),
                    "bbox_2d": [x1, y1, x2, y2]
                })
        
        return result
    
    def plot_bounding_boxes(self, im, model_response):
        """绘制边界框"""
        img = im.copy()
        width, height = img.size
        
        draw = ImageDraw.Draw(img)
        colors = additional_colors
        
        try:
            font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
        except:
            font = ImageFont.load_default()
        
        # 解析Youtu VL格式的响应
        bounding_boxes = self.parse_youtu_vl_response(model_response)
        
        # 如果没有检测到Youtu VL格式，尝试其他格式
        if not bounding_boxes:
            # 尝试匹配更宽松的模式
            relaxed_pattern = r'<box>(.*?)</box>'
            box_matches = re.findall(relaxed_pattern, model_response, re.DOTALL)
            
            for i, box_content in enumerate(box_matches):
                box_pattern = r'<x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)>'
                box_match = re.search(box_pattern, box_content)
                
                if box_match:
                    x1, y1, x2, y2 = map(int, box_match.groups())
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                        
                    bounding_boxes.append({
                        "label": f"obj_{i+1}",
                        "bbox_2d": [x1, y1, x2, y2]
                    })
        
        # 绘制边界框
        for i, bbox_info in enumerate(bounding_boxes):
            if "bbox_2d" not in bbox_info:
                continue
            
            bbox = bbox_info["bbox_2d"]
            if len(bbox) < 4:
                continue
            
            color = colors[i % len(colors)]
            
            # 获取坐标
            x1, y1, x2, y2 = bbox[:4]
            
            # 绘制边界框
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
            
            # 绘制标签
            label = bbox_info.get("label", f"obj_{i+1}")
            # 添加白色背景使标签更清晰
            text_bbox = draw.textbbox((x1 + 8, y1 + 6), label, font=font)
            draw.rectangle(text_bbox, fill="white")
            draw.text((x1 + 8, y1 + 6), label, fill=color, font=font)
        
        return img
    
    def process_image_with_model(self, image, prompt):
        """处理图像和提示词"""
        try:
            # 如果是PIL图像，先保存到临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name)
                img_path = tmp_file.name
            
            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # 准备输入
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # 生成响应
            generated_ids = self.model.generate(
                **inputs,
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                do_sample=False,  # 设置为False以获得确定性输出
                max_new_tokens=1024,
                img_input=img_path,  # Youtu-VL需要的特殊参数
            )
            
            # 解码输出
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            generated_text = outputs[0]
            
            # 清理临时文件
            import os
            os.unlink(img_path)
            
            # 判断是否为检测任务
            result_image = image.copy()
            detection_keywords = ['检测', 'detect', 'bounding box', 'bbox', 'box', 'object detection']
            prompt_lower = prompt.lower()
            
            if any(keyword in prompt_lower for keyword in detection_keywords) or "<box>" in generated_text:
                result_image = self.plot_bounding_boxes(image, generated_text)
            
            return generated_text, result_image
            
        except Exception as e:
            import traceback
            error_msg = f"处理过程中出现错误: {str(e)}\n{traceback.format_exc()}"
            return error_msg, image


# 创建Gradio界面
def create_gradio_interface(model_handler):
    with gr.Blocks(title="Youtu-VL 图像理解演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Youtu-VL 图像理解演示
        上传图像并输入提示词，模型会生成回复。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil", 
                    label="上传图像",
                    height=400
                )
                prompt_input = gr.Textbox(
                    label="提示词",
                    placeholder="例如：检测图像中的所有物体 或 描述这张图像",
                    lines=3
                )
                submit_btn = gr.Button("提交", variant="primary", size="lg")
                
                # 示例提示词
                examples = gr.Examples(
                    examples=[
                        ["描述这张图像。", "描述图像内容"],
                        ["检测图像中的所有物体。", "物体检测"],
                        ["这是什么场景？", "场景理解"],
                        ["图像中有哪些物体？", "物体识别"],
                        ["详细描述图像内容。", "详细描述"],
                    ],
                    inputs=[prompt_input, gr.Textbox(visible=False)],
                    label="示例提示词"
                )
            
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("模型回复"):
                        text_output = gr.Textbox(
                            label="模型回复", 
                            lines=15,
                            show_copy_button=True
                        )
                    with gr.TabItem("可视化结果"):
                        image_output = gr.Image(
                            label="检测结果可视化",
                            height=400
                        )
        
        # 提交按钮事件
        def process_inputs(image, prompt):
            if image is None:
                return "请上传图像", None
            return model_handler.process_image_with_model(image, prompt)
        
        submit_btn.click(
            fn=process_inputs,
            inputs=[image_input, prompt_input],
            outputs=[text_output, image_output]
        )
        
        # 使用说明
        gr.Markdown("""
        ## 使用说明
        
        ### 检测任务
        - 提示词示例：`检测图像中的所有物体`
        - 模型会返回物体的边界框坐标
        - 自动在图像上绘制边界框和标签
        
        ### 描述任务
        - 提示词示例：`描述这张图像`
        - 模型会生成详细的图像描述
        
        ### 注意事项
        1. 确保图像清晰
        2. 检测任务需要较长的处理时间
        3. 模型支持中文和英文提示词
        """)
    
    return demo


if __name__ == "__main__":
    # 初始化模型
    model_handler = YoutuVLModel()
    
    # 创建界面
    demo = create_gradio_interface(model_handler)
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=9009,
        share=False,
        debug=False
    )