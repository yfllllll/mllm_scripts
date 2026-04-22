import os
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
from vllm import LLM, SamplingParams
import time

# 环境变量设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["GRADIO_TEMP_DIR"] = "/mnt/data/lyf/tmp"

# 初始化模型
def run_qwen2_5_vl():
    model_name = "/mnt/data/lyf/qwenvl-2.5-72b"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
    )
    return llm

print("正在加载模型...")
llm = run_qwen2_5_vl()
print("模型加载完成!")

# 绘制检测框函数
def draw_boxes_on_image(image, boxes, color="red", font_path="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size=20):
    """
    在图像上绘制检测框和标签，支持中文。
    """
    # 如果image是numpy数组，转换为PIL Image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image.copy()
    
    draw = ImageDraw.Draw(image_pil)

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"字体文件未找到: {font_path}. 使用默认字体.")
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
    从响应文本中解析检测框坐标。
    """
    # 正则表达式匹配检测框
    box_pattern = r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"(?:,\s*"sub lable":\s*"(.*?)")?'
    
    matches = re.findall(box_pattern, response)
    
    # 解析每一个框
    category_boxes = {}
    for match in matches:
        x1, y1, x2, y2, class_name, sub_classname = match
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        if sub_classname:
            class_name = class_name + '/' + sub_classname
        if class_name not in category_boxes:
            category_boxes[class_name] = []
        
        category_boxes[class_name].append([x1, y1, x2, y2])

    # 生成格式化后的检测框列表
    boxes = []
    for category, box_list in category_boxes.items():
        for box in box_list:
            boxes.append((category, box))
    return boxes

# 处理图像并推理
def process(image, category_name):
    """
    处理上传的图像并进行目标检测
    """
    # 如果图像是None，直接返回错误
    if image is None:
        return "请上传一张图像", None
    
    try:
        # 确保图像是PIL格式
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, str):  # 文件路径
            pil_image = Image.open(image).convert("RGB")
        else:  # 假设已经是PIL图像
            pil_image = image.convert("RGB")
        
        # 构建推理请求
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
            stop=["<|im_end|>"]
        )
        
        placeholder = "<|image_pad|>"
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                  f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                  f"{category_name}<|im_end|>\n"
                  "<|im_start|>assistant\n")
        
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {'image': pil_image},
        }

        # 使用 vLLM 进行推理
        outputs = llm.generate([inputs], sampling_params)
        output_text = outputs[0].outputs[0].text
        print("模型输出:", output_text)
        
        # 解析检测框
        parsed_boxes = parse_response_boxes(output_text)
        
        # 绘制检测框
        if parsed_boxes:
            result_image = draw_boxes_on_image(pil_image, parsed_boxes, color="blue")
        else:
            result_image = pil_image
        
        return output_text, result_image
        
    except Exception as e:
        print(f"处理错误: {e}")
        return f"处理过程中出现错误: {str(e)}", None

# 创建Gradio界面
with gr.Blocks(title="Qwen2.5-VL 目标检测") as demo:
    gr.Markdown("# Qwen2.5-VL 目标检测系统")
    
    with gr.Row():
        with gr.Column():
            # 使用标准的Gradio图像组件
            image_input = gr.Image(
                label="上传图像",
                type="pil",  # 直接使用PIL图像，避免numpy转换问题
                height=400
            )
            category_input = gr.Textbox(
                label="检测指令", 
                placeholder="例如: 检测图像中的所有车辆和行人",
                value="检测图像中的目标物体并给出边界框坐标",
                lines=2
            )
            submit_btn = gr.Button("开始检测", variant="primary")
            
            # 添加说明
            gr.Markdown("""
            ### 使用说明:
            1. 上传一张图像
            2. 输入检测指令（可选）
            3. 点击"开始检测"按钮
            4. 查看检测结果和模型响应
            """)
        
        with gr.Column():
            response_output = gr.Textbox(
                label="模型响应", 
                lines=10,
                max_lines=15
            )
            result_image = gr.Image(
                label="检测结果", 
                type="pil",
                height=400
            )
    
    # 连接按钮事件
    submit_btn.click(
        fn=process,
        inputs=[image_input, category_input],
        outputs=[response_output, result_image]
    )
    
    # 添加示例
    gr.Examples(
        examples=[
            ["检测图像中的所有人", "检测图像中的所有人和车辆"],
            ["找出图像中的所有建筑", "检测图像中的所有建筑物"],
        ],
        inputs=[category_input],
        label="示例指令"
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=9005,
        share=False,
        show_error=True,
        debug=True
    )