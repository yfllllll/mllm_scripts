from transformers import AutoModelForImageTextToText, AutoProcessor
import json
import random
import io
import ast
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import gradio as gr
import torch

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

# 加载模型和处理器（全局加载一次）
model = AutoModelForImageTextToText.from_pretrained(
    "/mnt/data/lyf/Qwen3-VL-235B-A22B-Instruct",  
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("/mnt/data/lyf/Qwen3-VL-235B-A22B-Instruct")

def decode_json_points(text: str):
    """Parse coordinate points from text format"""
    try:
        # 清理markdown标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        
        # 解析JSON
        data = json.loads(text)
        points = []
        labels = []
        
        for item in data:
            if "point_2d" in item:
                x, y = item["point_2d"]
                points.append([x, y])
                
                # 获取label，如果没有则使用默认值
                label = item.get("label", f"point_{len(points)}")
                labels.append(label)
        
        return points, labels
        
    except Exception as e:
        print(f"Error: {e}")
        return [], []

def parse_json(json_output):
    """Parsing out the markdown fencing"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def plot_bounding_boxes(im, bounding_boxes):
    """Plots bounding boxes on an image"""
    img = im.copy()
    width, height = img.size
    
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
        'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
        'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    try:
        font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
    except:
        font = ImageFont.load_default()

    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        try:
            # 尝试直接解析JSON
            json_output = json.loads(bounding_boxes)
        except:
            # 如果还是失败，尝试截断处理
            end_idx = bounding_boxes.rfind('"}') + len('"}')
            truncated_text = bounding_boxes[:end_idx] + "]"
            try:
                json_output = ast.literal_eval(truncated_text)
            except:
                # 最终失败，返回原图
                return img

    if not isinstance(json_output, list):
        json_output = [json_output]

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
        # Select a color from the list
        color = colors[i % len(colors)]

        # 检查是否有bbox_2d字段
        if "bbox_2d" not in bounding_box:
            continue

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3
        )

        # Draw the text
        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    return img

def plot_points(im, text):
    """Plots points on an image"""
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
        'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
        'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors

    points, descriptions = decode_json_points(text)
    
    if not points:
        return img

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except:
        font = ImageFont.load_default()

    for i, point in enumerate(points):
        color = colors[i % len(colors)]
        abs_x1 = int(point[0])/1000 * width
        abs_y1 = int(point[1])/1000 * height
        radius = 5
        draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
        if i < len(descriptions):
            draw.text((abs_x1 - 20, abs_y1 + 6), descriptions[i], fill=color, font=font)
    
    return img

def process_image_with_model(image, prompt):
    """处理图像和提示词，返回模型回复和可视化结果"""
    try:
        # 准备消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 准备输入
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # 生成回复
        generated_ids = model.generate(**inputs, max_new_tokens=9280)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 根据输出内容决定如何可视化
        result_image = image.copy()
        
        # 检查是否是检测任务（包含bbox_2d或point_2d）
        if "bbox_2d" in output_text:
            result_image = plot_bounding_boxes(image, output_text)
        elif "point_2d" in output_text:
            result_image = plot_points(image, output_text)
        
        return output_text, result_image
        
    except Exception as e:
        error_msg = f"处理过程中出现错误: {str(e)}"
        return error_msg, image

# 创建Gradio界面
with gr.Blocks(title="Qwen-VL 图像理解演示") as demo:
    gr.Markdown("""
    # Qwen-VL 图像理解演示
    上传图像并输入提示词，模型会生成回复。如果是检测任务，结果会直接绘制在图像上。
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传图像")
            prompt_input = gr.Textbox(
                label="提示词",
                placeholder="例如：检测图像中的所有物体并给出边界框,以 JSON 格式报告边界框坐标",
                lines=3
            )
            submit_btn = gr.Button("提交", variant="primary")
            
            # 示例提示词
            examples = gr.Examples(
                examples=[
                    ["描述这张图像。", "描述图像内容"],
                    ["检测图像中的所有物体。", "物体检测"],
                    ["图像中有哪些显著的点？", "点检测"],
                    ["这是什么场景？", "场景理解"]
                ],
                inputs=[prompt_input, gr.Textbox(visible=False)],
                label="示例提示词"
            )
        
        with gr.Column():
            text_output = gr.Textbox(label="模型回复", lines=10)
            image_output = gr.Image(label="可视化结果")
    
    # 提交按钮事件
    submit_btn.click(
        fn=process_image_with_model,
        inputs=[image_input, prompt_input],
        outputs=[text_output, image_output]
    )
    
    # 添加一些使用说明
    gr.Markdown("""
    ## 使用说明
    - **检测任务提示词**: 使用如"检测物体"、"找出边界框"、"标记关键点"等提示词
    - **描述任务**: 使用如"描述图像"、"这是什么"等提示词
    - **最佳实践**: 提示词越具体，模型回复越准确
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=9005)
