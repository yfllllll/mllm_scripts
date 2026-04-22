# -*- coding: utf-8 -*-
import os
import requests
import json
import ast
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# ====================== 配置 ======================
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8002/v1")  # vLLM API 地址
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "QwenVl")              # 模型名称
TARGET_WIDTH = 1280  # 720p 宽度
TARGET_HEIGHT = 720  # 720p 高度

# 颜色定义
additional_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
                    'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta', 
                    'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 
                    'lavender', 'violet', 'gold', 'silver']

# ====================== 辅助函数 ======================
def resize_to_720p(image: Image.Image) -> Image.Image:
    """
    将图像等比缩放到适配720p（1280×720）画布，居中放置在黑色背景上
    参数：
        image: 输入的PIL Image对象
    返回：
        尺寸为1280×720的PIL Image对象
    """
    # 1. 获取原图像尺寸
    original_width, original_height = image.size
    
    # 2. 计算等比缩放比例（保证图像完整显示在720p画布内）
    scale_w = TARGET_WIDTH / original_width
    scale_h = TARGET_HEIGHT / original_height
    scale = min(scale_w, scale_h)  # 取较小的缩放比例，避免图像超出画布
    
    # 3. 计算缩放后的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 4. 等比缩放图像
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 5. 创建1280×720的黑色画布，将缩放后的图像居中放置
    canvas = Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0))
    offset_x = (TARGET_WIDTH - new_width) // 2
    offset_y = (TARGET_HEIGHT - new_height) // 2
    canvas.paste(resized_image, (offset_x, offset_y))
    
    return canvas

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
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1,
        "stream": False
    }
    try:
        response = requests.post(f"{VLLM_API_URL}/chat/completions", json=payload, headers=headers, timeout=90)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
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
    # 先将图像缩放到720p（保证绘制尺寸和模型输入一致）
    img = resize_to_720p(im.copy())
    width, height = img.size  # 此时宽高固定为1280×720
    
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

        # 720p画布下直接使用千分比计算坐标（宽1280=1000份，高720=1000份）
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
    # 先将图像缩放到720p
    img = resize_to_720p(im.copy())
    width, height = img.size  # 1280×720
    
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
            print(f"解析点坐标错误: {e}")
            return [], []

    points, descriptions = decode_json_points(text)
    
    if not points:
        return img

    try:
        font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
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

def extract_point_info(point_text):
    """从point-based grounding结果中提取点信息（用于构造新提示词）"""
    try:
        point_text = parse_json(point_text)
        data = json.loads(point_text) if "{" in point_text else ast.literal_eval(point_text)
        point_info = []
        if not isinstance(data, list):
            data = [data]
        for item in data:
            if "point_2d" in item and "label" in item:
                point_info.append(f"{item['label']} (坐标: {item['point_2d'][0]}, {item['point_2d'][1]})")
        return "; ".join(point_info) if point_info else "未检测到有效关键点"
    except Exception as e:
        print(f"提取点信息失败: {e}")
        return "关键点信息解析失败"

class QwenVLvLLM:
    """Qwen-VL vLLM API 客户端类"""
    def __init__(self):
        # 无需加载模型，直接通过 API 调用
        pass

    def process_image_with_model(self, image, prompt):
        """原始模式：直接处理图像和提示词，调用 vLLM API"""
        try:
            # 第一步：将图像缩放到720p规格
            resized_image = resize_to_720p(image)
            
            # 第二步：转换为base64
            img_base64 = pil_to_base64(resized_image)
            img_url = f"data:image/jpeg;base64,{img_base64}"

            # 第三步：构造消息并调用API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            generated_text = call_vllm_chat(messages, temperature=0, max_tokens=9280)

            # 第四步：可视化结果（绘制在720p图像上）
            result_image = image.copy()  # 保留原图用于展示，但绘制在720p版本上
            if "bbox_2d" in generated_text:
                result_image = plot_bounding_boxes(image, generated_text)
            elif "point_2d" in generated_text:
                result_image = plot_points(image, generated_text)

            return generated_text, result_image

        except Exception as e:
            error_msg = f"处理过程中出现错误: {str(e)}"
            return error_msg, image

    def guided_detection_pipeline(self, image, prompt):
        """指引式检测Pipeline：先Point-based Grounding → 绘制点 → 目标检测"""
        try:
            # ========== 第一步：图像缩放到720p ==========
            resized_image = resize_to_720p(image)
            
            # ========== 第二步：Point-based Grounding ==========
            ground_prompt = "请根据用户输入" + prompt + """，对图像进行point-based grounding，定位需要关注的关键区域，
            并以JSON格式输出每个关键点的point_2d坐标和对应的label，
            输出格式要求：[{"point_2d":[x,y],"label":"xxx"}, ...]"""
            
            # 调用模型做point grounding（传入720p图像）
            img_base64 = pil_to_base64(resized_image)
            img_url = f"data:image/jpeg;base64,{img_base64}"
            ground_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": ground_prompt}
                    ]
                }
            ]
            point_result = call_vllm_chat(ground_messages, temperature=0, max_tokens=9280)
            
            # 绘制关键点（在720p图像上）
            image_with_points = plot_points(image, point_result)  # 基于原图绘制720p结果
            
            # 提取点信息
            point_info = extract_point_info(point_result)
            
            # ========== 第三步：带指引的目标检测 ==========
            guided_prompt = "基于以下关键点信息：" + point_info + """
            """ + prompt + """
            请以JSON格式输出目标的bbox_2d边界框和对应的label，
            输出格式要求：[{"bbox_2d":[x1,y1,x2,y2],"label":"xxx"}, ...]"""
            
            # 再次缩放带关键点的图像（确保是720p）
            resized_with_points = resize_to_720p(image_with_points)
            img_with_points_base64 = pil_to_base64(resized_with_points)
            img_with_points_url = f"data:image/jpeg;base64,{img_with_points_base64}"
            
            # 调用目标检测
            detect_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_with_points_url}},
                        {"type": "text", "text": guided_prompt}
                    ]
                }
            ]
            detect_result = call_vllm_chat(detect_messages, temperature=0, max_tokens=9280)
            
            # 绘制最终边界框（720p）
            final_image = plot_bounding_boxes(image_with_points, detect_result)
            
            # 拼接结果
            full_result = f"【Point-based Grounding结果】:\n{point_result}\n\n【指引式检测结果】:\n{detect_result}"
            return full_result, final_image

        except Exception as e:
            error_msg = f"指引式Pipeline处理失败: {str(e)}"
            return error_msg, image

# ====================== Gradio 界面 ======================
def create_gradio_interface(model_handler):
    with gr.Blocks(title="Qwen-VL vLLM API 图像理解演示") as demo:
        gr.Markdown("""
        # Qwen-VL vLLM API 图像理解演示
        通过 vLLM API 进行高效推理，支持两种检测模式切换：
        - 原始模式：直接输入提示词进行检测
        - 指引式模式：先Point-based Grounding定位关键点，再做目标检测
        > 注：所有图像会自动缩放到720p（1280×720）规格后处理
        """)
        
        # 模式选择
        with gr.Row():
            mode_radio = gr.Radio(
                choices=["原始模式", "指引式模式"],
                value="原始模式",
                label="检测模式选择",
                interactive=True
            )
        
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
                image_output = gr.Image(label="可视化结果（720p）")
        
        # 提交按钮事件
        def process_inputs(mode, image, prompt):
            if image is None:
                return "请上传图像", None
            if mode == "原始模式":
                return model_handler.process_image_with_model(image, prompt)
            elif mode == "指引式模式":
                return model_handler.guided_detection_pipeline(image, prompt)
        
        submit_btn.click(
            fn=process_inputs,
            inputs=[mode_radio, image_input, prompt_input],
            outputs=[text_output, image_output]
        )
        
        # 使用说明
        gr.Markdown("""
        ## 使用说明
        - **检测任务**: 使用如"检测物体,以 JSON 格式报告边界框坐标"等提示词
        - **文字识别**: 使用如"读取文字"、"识别文本"等提示词
        - **图像规格**: 所有上传图像会自动缩放到720p（1280×720）后处理
        - **指引式模式**: 先定位关键点，再基于关键点做目标检测，提升准确性
        """)
    
    return demo

if __name__ == "__main__":
    print("正在连接到 vLLM API 服务...")
    # 检查 API 可用性
    try:
        r = requests.get(f"{VLLM_API_URL}/models", timeout=5)
        if r.status_code == 200:
            models = r.json().get("data", [])
            print(f"可用模型: {[m['id'] for m in models]}")
        else:
            print("警告: 无法获取模型列表，请确保 vLLM 服务已启动")
    except Exception as e:
        print(f"警告: 连接到 vLLM API 失败: {e}")
    
    model_handler = QwenVLvLLM()
    print("API 客户端初始化完成！")
    
    demo = create_gradio_interface(model_handler)
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=9005,
        share=False
    )