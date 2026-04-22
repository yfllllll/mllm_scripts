# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import gradio as gr
from gradio_image_annotation import image_annotator
from PIL import Image, ImageDraw, ImageFont
import json
import ast
import tempfile

# 设置环境变量
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# 颜色定义
additional_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                    (255, 165, 0), (255, 192, 203), (128, 0, 128)]

def prepare_inputs_for_vllm(messages, processor):
    """准备vLLM输入"""
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

def draw_boxes_on_image(image, boxes_data):
    """在图像上绘制矩形框"""
    if image is None:
        return None
        
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    try:
        font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
    except:
        font = ImageFont.load_default()
    
    if not isinstance(boxes_data, list):
        boxes_data = [boxes_data]
    
    for i, box in enumerate(boxes_data):
        color = additional_colors[i % len(additional_colors)]
        
        # 从标注数据中获取坐标
        xmin = box["xmin"]
        ymin = box["ymin"] 
        xmax = box["xmax"]
        ymax = box["ymax"]
        
        # 绘制矩形框
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        # 添加标签
        label = f"Region {i+1}"
        draw.text((xmin + 5, ymin + 5), label, fill=color, font=font)
    
    return img

def format_boxes_for_prompt(boxes_data, image_width, image_height):
    """将框坐标格式化为模型提示词"""
    formatted_boxes = []
    
    if not isinstance(boxes_data, list):
        boxes_data = [boxes_data]
    
    for i, box in enumerate(boxes_data):
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"] 
        ymax = box["ymax"]
        
        # 转换为归一化坐标 [0, 1000]
        norm_x1 = int(xmin * 1000 / image_width)
        norm_y1 = int(ymin * 1000 / image_height)
        norm_x2 = int(xmax * 1000 / image_width)
        norm_y2 = int(ymax * 1000 / image_height)
        
        formatted_boxes.append(
            f'区域 {i+1} (坐标：<{norm_x1}><{norm_y1}><{norm_x2}><{norm_y2}>): '
        )

    return formatted_boxes

class QwenVLvLLM:
    """Qwen-VL vLLM推理类"""
    def __init__(self, checkpoint_path="/mnt/data/lyf/Qwen3-VL-32B-Instruct"):
        self.checkpoint_path = checkpoint_path
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        
        # 初始化vLLM模型
        self.llm = LLM(
            model=checkpoint_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=False,
            max_model_len=41960,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
            dtype="bfloat16",
            tokenizer_mode='slow'
        )
        
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=9280,
            top_k=-1,
            stop_token_ids=[],
        )
    
    def describe_regions(self, annotation_data, custom_prompt=None):
        """描述指定区域的内容"""
        # try:
        if not annotation_data or "image" not in annotation_data:
            return "", "请先上传图像并绘制区域", None
        
        image = annotation_data["image"]
        boxes_data = annotation_data.get("boxes", [])
        
        if not boxes_data:
            return "", "请在图像上绘制矩形框", image
        
        width, height = image.size
        


        # 格式化框数据
        formatted_boxes = format_boxes_for_prompt(boxes_data, width, height)
        
        # 构建区域描述提示词
        if custom_prompt:
            regions_text = custom_prompt
        else:
            regions_text = "请用短语描述每一个区域主体，格式为region id: brief descripted instance。\n\n"
            for i, box in enumerate(formatted_boxes):
                x1, y1, x2, y2 = box["bbox_2d"]
                regions_text += f"区域 {i+1}" f"(坐标: <{x1}><{y1}><{x2}><{y2}>): "

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": ''},
                ],
            }
        ]
        
        # 添加框信息到消息中
        if formatted_boxes:
            messages[0]["content"].append({
                "type": "text", 
                "text": f"请回答以下关于图像中某些区域的问题：\n{formatted_boxes[0]} {custom_prompt}?"
            })
        
        # 获取输入提示词
        input_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = [prepare_inputs_for_vllm(messages, self.processor)]
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        description = outputs[0].outputs[0].text
        
        # 在图像上绘制框
        annotated_image = draw_boxes_on_image(image, boxes_data)
        
        return input_prompt, description, annotated_image
            
        # except Exception as e:
        #     error_msg = f"区域描述过程中出现错误: {str(e)}"
        #     return "", error_msg, None

# 创建Gradio界面
def create_gradio_interface(model_handler):
    with gr.Blocks(title="Qwen-VL 区域描述演示") as demo:
        gr.Markdown("""
        # Qwen-VL 区域描述演示
        使用专门的图像标注工具绘制矩形框，模型将描述指定区域的内容。
        """)
        
        with gr.Row():
            with gr.Column():
                # 使用专业的图像标注组件[citation:1][citation:4]
                annotator_component = image_annotator(
                    None,
                    label_list=["区域1", "区域2", "区域3", "区域4", "区域5"],
                    label_colors=additional_colors[:5],
                    box_thickness=3,
                    handle_size=8,
                    use_default_label=True,
                    image_type="pil",
                    boxes_alpha=0
                )
                
                # 自定义提示词
                custom_prompt = gr.Textbox(
                    label="自定义提示词 (可选)",
                    placeholder="例如：描述这些区域中的物体，或者分析区域之间的关系",
                    lines=2
                )
                
                with gr.Row():
                    clear_btn = gr.Button("清除所有框", variant="secondary")
                    describe_btn = gr.Button("描述区域", variant="primary")
                
                # 手动坐标输入
                with gr.Accordion("手动输入区域坐标", open=False):
                    gr.Markdown("输入JSON格式的坐标（基于图像实际像素尺寸）：")
                    custom_boxes = gr.Textbox(
                        label="区域坐标",
                        placeholder='[{"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200, "label": "区域1"}]',
                        lines=3
                    )
                    apply_custom_btn = gr.Button("应用自定义坐标")
            
            with gr.Column():
                # 输入提示词显示
                input_prompt_display = gr.Textbox(
                    label="输入提示词",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    show_copy_button=True
                )
                
                # 模型输出显示
                text_output = gr.Textbox(
                    label="模型输出",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                
                # 可视化结果
                image_output = gr.Image(label="标注结果", height=400)
        
        def describe_regions(annotation_data, prompt):
            """描述区域内容"""
            if not annotation_data or "image" not in annotation_data:
                return "", "请先上传图像", None
            
            return model_handler.describe_regions(annotation_data, prompt)
        
        def clear_annotation():
            """清除标注"""
            return None
        
        def apply_custom_boxes(annotation_data, custom_text):
            """应用自定义坐标"""
            try:
                if not annotation_data or "image" not in annotation_data:
                    return "", "请先上传图像", None
                
                new_boxes = json.loads(custom_text)
                if not isinstance(new_boxes, list):
                    new_boxes = [new_boxes]
                
                # 更新标注数据
                updated_annotation = {
                    "image": annotation_data["image"],
                    "boxes": new_boxes
                }
                
                # 绘制标注图像
                annotated_image = draw_boxes_on_image(annotation_data["image"], new_boxes)
                
                return "", "自定义坐标应用成功", annotated_image
                
            except Exception as e:
                return "", f"坐标格式错误: {str(e)}", None
        
        # 事件绑定
        describe_btn.click(
            fn=describe_regions,
            inputs=[annotator_component, custom_prompt],
            outputs=[input_prompt_display, text_output, image_output]
        )
        
        clear_btn.click(
            fn=clear_annotation,
            inputs=[],
            outputs=[annotator_component]
        )
        
        apply_custom_btn.click(
            fn=apply_custom_boxes,
            inputs=[annotator_component, custom_boxes],
            outputs=[input_prompt_display, text_output, image_output]
        )
        
        # 使用说明
        gr.Markdown("""
        ## 使用说明
        
        ### 方法一：交互式绘制
        1. **上传图像**：点击标注区域上传按钮选择图像
        2. **绘制矩形框**：
           - 在图像上按住鼠标左键并拖动来绘制矩形框
           - 可以绘制多个矩形框
           - 绘制完成后可以拖动调整位置或大小
        3. **添加标签**：可以为每个框选择或输入标签
        4. **获取描述**：点击"描述区域"按钮
        
        ### 方法二：手动输入坐标
        1. 上传图像
        2. 展开"手动输入区域坐标"面板  
        3. 输入JSON格式的坐标，例如：
           ```json
           [
             {"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200, "label": "区域1"},
             {"xmin": 300, "ymin": 150, "xmax": 400, "ymax": 250, "label": "区域2"}
           ]
           ```
        4. 点击"应用自定义坐标"
        5. 点击"描述区域"按钮
        
        ### 绘制技巧
        - **开始绘制**：按住鼠标左键并拖动
        - **调整位置**：拖动框的内部
        - **调整大小**：拖动框边缘的控制点
        - **删除框**：选中框后按Delete键或使用右键菜单
        
        ### 显示说明
        - **输入提示词**：显示发送给模型的完整提示词
        - **模型输出**：显示模型的原始输出文本  
        - **标注结果**：显示带有区域框的可视化图像
        """)
    
    return demo

if __name__ == "__main__":
    # 初始化模型
    print("正在加载模型...")
    model_handler = QwenVLvLLM("/mnt/disk/lyf/ms-swift/output/rex_omni_labelme_full_data1_data2_support_neg_imgcap_improved/v9-20260126-204156/checkpoint-18000")
    print("模型加载完成！")
    
    # 创建并启动界面
    demo = create_gradio_interface(model_handler)
    demo.launch(
        server_name="0.0.0.0",
        server_port=9008,
        share=False
    )