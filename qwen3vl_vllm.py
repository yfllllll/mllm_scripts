# -*- coding: utf-8 -*-
import os
from openai import OpenAI
import json
import ast
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import gradio as gr



# 颜色定义
additional_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
                    'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta', 
                    'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 
                    'lavender', 'violet', 'gold', 'silver']

# ====================== 辅助函数 ======================
def pil_to_base64(image: Image.Image) -> str:
    """将 PIL Image 转换为 base64 字符串（JPEG 格式）"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def call_vllm_chat(messages: list, temperature: float = 0, max_tokens: int = 9280) -> str:
    """
    调用 vLLM 的聊天补全 API
    参数：
        messages: 符合 OpenAI 格式的消息列表
        temperature: 采样温度
        max_tokens: 最大生成 token 数
    返回：
        模型生成的文本内容
    """
    client = OpenAI(
        base_url="http://localhost:9005/v1",
        api_key="EMPTY"
    )

    try:
        response = client.chat.completions.create(
            model="gemma4",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"vLLM API 调用失败: {e}")

def parse_json(json_output):
    """解析JSON输出（去除可能的 markdown 代码块）"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def plot_bounding_boxes(im, bounding_boxes):
    """绘制边界框"""
    img = im.copy()
    width, height = img.size
    
    draw = ImageDraw.Draw(img)
    colors = additional_colors

    bounding_boxes = parse_json(bounding_boxes)

    try:
        font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
    except:
        font = ImageFont.load_default()

    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        try:
            json_output = json.loads(bounding_boxes)
        except:
            try:
                end_idx = bounding_boxes.rfind('"}') + len('"}')
                truncated_text = bounding_boxes[:end_idx] + "]"
                json_output = ast.literal_eval(truncated_text)
            except:
                return img

    if not isinstance(json_output, list):
        json_output = [json_output]

    for i, bounding_box in enumerate(json_output):
        color = colors[i % len(colors)]

        if "bbox_2d" not in bounding_box:
            continue

        abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)

        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    return img

def plot_points(im, text):
    """绘制点"""
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    def decode_json_points(text: str):
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            
            data = json.loads(text)
            points = []
            labels = []
            
            for item in data:
                if "point_2d" in item:
                    x, y = item["point_2d"]
                    points.append([x, y])
                    label = item.get("label", f"point_{len(points)}")
                    labels.append(label)
            
            return points, labels
            
        except Exception as e:
            print(f"Error: {e}")
            return [], []

    points, descriptions = decode_json_points(text)
    
    if not points:
        return img

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except:
        font = ImageFont.load_default()

    for i, point in enumerate(points):
        color = additional_colors[i % len(additional_colors)]
        abs_x1 = int(point[0]) / 1000 * width
        abs_y1 = int(point[1]) / 1000 * height
        radius = 5
        draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
        if i < len(descriptions):
            draw.text((abs_x1 - 20, abs_y1 + 6), descriptions[i], fill=color, font=font)
    
    return img

class QwenVLvLLM:
    """Qwen-VL vLLM API 客户端类"""
    def __init__(self):
        # 无需加载模型，直接通过 API 调用
        pass

    def process_image_with_model(self, image, prompt):
        """处理图像和提示词，调用 vLLM API"""
        try:
            # 将 PIL 图像转换为 base64 data URL
            img_base64 = pil_to_base64(image)
            img_url = f"data:image/jpeg;base64,{img_base64}"

            # 构造消息（OpenAI 格式）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 调用 API
            generated_text = call_vllm_chat(messages, temperature=0, max_tokens=9280)

            # 根据输出内容决定如何可视化
            result_image = image.copy()
            if "bbox_2d" in generated_text:
                result_image = plot_bounding_boxes(image, generated_text)
            elif "point_2d" in generated_text:
                result_image = plot_points(image, generated_text)

            return generated_text, result_image

        except Exception as e:
            error_msg = f"处理过程中出现错误: {str(e)}"
            return error_msg, image

# ====================== Gradio 界面 ======================
def create_gradio_interface(model_handler):
    with gr.Blocks(title="Qwen-VL vLLM API 图像理解演示") as demo:
        gr.Markdown("""
        # Qwen-VL vLLM API 图像理解演示
        通过 vLLM API 进行高效推理。上传图像并输入提示词，模型会生成回复。
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图像")
                prompt_input = gr.Textbox(
                    label="提示词",
                    placeholder="例如：检测图像中的所有物体并给出边界框",
                    lines=3
                )
                submit_btn = gr.Button("提交", variant="primary")
                
                # 示例提示词
                examples = gr.Examples(
                    examples=[
                        ["描述这张图像。", "描述图像内容"],
                        ["检测图像中的所有物体,以 JSON 格式报告边界框坐标。", "物体检测"],
                        ["图像中有哪些显著的点？", "点检测"],
                        ["这是什么场景？", "场景理解"],
                        ["读取图像中的所有文字。", "文字识别"]
                    ],
                    inputs=[prompt_input, gr.Textbox(visible=False)],
                    label="示例提示词"
                )
            
            with gr.Column():
                text_output = gr.Textbox(label="模型回复", lines=10)
                image_output = gr.Image(label="可视化结果")
        
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
        - **检测任务**: 使用如"检测物体,以 JSON 格式报告边界框坐标"、"找出边界框,以 JSON 格式报告边界框坐标"等提示词
        - **文字识别**: 使用如"读取文字"、"识别文本"等提示词
        - **描述任务**: 使用如"描述图像"、"这是什么"等提示词
        - **后端**: 通过 vLLM API 调用，支持多模态输入
        """)
    
    return demo

if __name__ == "__main__":
    print("正在连接到 vLLM API 服务...")
    # 检查 API 是否可用（可选）
    # try:
    #     r = requests.get(f"{VLLM_API_URL}/models", timeout=5)
    #     if r.status_code == 200:
    #         models = r.json().get("data", [])
    #         print(f"可用模型: {[m['id'] for m in models]}")
    #     else:
    #         print("警告: 无法获取模型列表，请确保 vLLM 服务已启动")
    # except Exception as e:
    #     print(f"警告: 连接到 vLLM API 失败: {e}")
    
    model_handler = QwenVLvLLM()
    print("API 客户端初始化完成！")
    
    demo = create_gradio_interface(model_handler)
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=9006,
        share=False
    )
