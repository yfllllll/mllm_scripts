import os
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import ImageFont
import io

# 环境变量设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 模型初始化
weight_dir = '/mnt/data/lyf/qwenvl-2.5-72b'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    weight_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(weight_dir)

# 绘制检测框函数
def draw_boxes_on_image(image, boxes, color="red", font_path="./wqy-microhei.ttc", font_size=20):
    """
    在图像上绘制检测框和标签，支持中文。
    """
    # 如果输入是numpy数组，转换为PIL图像
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
        
    draw = ImageDraw.Draw(image_pil)

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font file not found at {font_path}. Using default font.")
        font = ImageFont.load_default()

    # 绘制每个检测框
    for class_name, box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{class_name}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_background = [x1, y1 - text_height, x1 + text_width, y1]
        draw.rectangle(text_background, fill=color)
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

    return image_pil

# 解析响应中的检测框
def parse_response_boxes(response):
    """
    从响应文本中解析检测框坐标，并且支持多个类别。
    """
    # 正则表达式解析框坐标和标签
    box_pattern = r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"(?:,\s*"sub lable":\s*"(.*?)")?'
    
    matches = re.findall(box_pattern, response[0])
    
    # 解析每一个框
    category_boxes = {}
    for match in matches:
        x1, y1, x2, y2, class_name, sub_classname = match
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # 检查类别是否已经在字典中，如果不在则创建一个空列表
        if sub_classname:
            class_name = class_name + '/' + sub_classname
        if class_name not in category_boxes:
            category_boxes[class_name] = []
        
        # 将框加入对应的类别
        category_boxes[class_name].append([x1, y1, x2, y2])

    # 生成格式化后的检测框列表
    boxes = []
    for category, box_list in category_boxes.items():
        for box in box_list:
            boxes.append((category, box))
    return boxes

# 将PIL图像转换为字节流，用于Gradio显示[1,3](@ref)
def pil_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# 处理图像并推理（完全在内存中处理）
def process(image_input, category_name):
    """
    完全在内存中处理图像，不保存任何临时文件[1](@ref)
    """
    # 将numpy数组转换为PIL图像[1](@ref)
    if isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input)
    else:
        pil_image = image_input

    # 构建推理请求
    text = category_name if category_name else ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},  # 直接传入PIL图像对象[1](@ref)
                {"type": "text", "text": text},
            ],
        }
    ]
    
    print("处理消息:", messages)

    try:
        # 模型推理
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 解析检测框
        print("模型输出:", output_text)
        parsed_boxes = parse_response_boxes(output_text)

        # 绘制检测框
        final_boxes = []
        if parsed_boxes:
            for box in parsed_boxes:
                final_boxes.append(box)
            # 直接在内存中处理图像，不保存文件[1](@ref)
            image_with_boxes = draw_boxes_on_image(pil_image, final_boxes, color="blue")
        else:
            image_with_boxes = pil_image

        # 返回文本结果和图像对象（Gradio会自动处理显示）[6,7](@ref)
        return output_text, image_with_boxes

    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        print(error_msg)
        return error_msg, image_input

# Gradio UI - 完全内存处理版本
with gr.Blocks() as demo:
    gr.Markdown("# 影像目标检测系统（无文件存储版本）")
    
    with gr.Row():
        with gr.Column():
            # 使用Gradio的图像组件直接上传和处理图像[6,7](@ref)
            image_input = gr.Image(
                label="上传影像", 
                type="numpy",  # 直接获取numpy数组格式[6](@ref)
                height=400
            )
            category_input = gr.Textbox(
                label="检测类别", 
                placeholder="请输入要检测的类别名称",
                info="例如：车辆, 建筑, 道路等"
            )
            submit_button = gr.Button("开始检测", variant="primary")
        
        with gr.Column():
            response_output = gr.Textbox(label="模型响应", lines=5)
            # 输出图像组件，Gradio会直接处理图像对象[6,7](@ref)
            result_image = gr.Image(
                label="检测结果", 
                type="pil",  # 直接处理PIL图像对象
                height=400
            )
    
    # 绑定事件
    submit_button.click(
        fn=process, 
        inputs=[image_input, category_input], 
        outputs=[response_output, result_image]
    )
    
    # 添加使用说明
    gr.Markdown("""
    ### 使用说明
    1. 上传一张待检测的影像
    2. 输入要检测的类别名称（多个类别用逗号分隔）
    3. 点击"开始检测"按钮
    4. 查看右侧的检测结果和模型响应
    
    **注意：** 本版本不会在本地保存任何临时文件，所有处理均在内存中完成。
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name='0.0.0.0', server_port=9005)