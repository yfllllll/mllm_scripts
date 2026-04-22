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
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tempfile

# 设置环境变量
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# 颜色定义
additional_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                    (255, 165, 0), (255, 192, 203), (128, 0, 128)]

query_class = "矩形框已在图中绘制出来，结合图像内容，判断矩形框的类别是猪马牛羊中的一种吗？"

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

def draw_visual_prompt(image, detections):
    """
    在图像上绘制半透明矩形框和坐标文字，作为视觉辅助
    """
    if image is None:
        return None
        
    # 转换到 RGBA 以支持透明度绘制
    img_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    
    try:
        # 尝试加载中文字体，如果不存在则使用默认
        font = ImageFont.truetype("./wqy-microhei.ttc", size=max(15, int(width/50)))
    except:
        font = ImageFont.load_default()
    
    for i, det in enumerate(detections):
        color = additional_colors[i % len(additional_colors)]
        x1, y1, x2, y2 = det['bbox']

        # 归一化坐标用于文字显示
        nx1, ny1 = int(x1 * 1000 / width), int(y1 * 1000 / height)
        nx2, ny2 = int(x2 * 1000 / width), int(y2 * 1000 / height)

        # 1. 绘制半透明填充和实线边框
        outline_color = color + (200,) # 较清晰的轮廓
        draw.rectangle([x1, y1, x2, y2], fill=None, outline=outline_color, width=3)
        
        # 2. 绘制标签背景和坐标文字
        label_text = f"Region {i+1}: <{nx1}><{ny1}><{nx2}><{ny2}>"
        # 获取文字大小以便绘制背景块
        try:
            text_bbox = draw.textbbox((x1, y1 - 25), label_text, font=font)
            draw.rectangle(text_bbox, fill=color + (180,)) # 文字底色提高对比度
        except:
            pass
        draw.text((x1, y1 - 25), label_text, fill=(255, 255, 255, 255), font=font)

    # 将覆盖层与原图合并
    return Image.alpha_composite(img_rgba, overlay).convert("RGB")

def _format_bbox_for_query(bbox):
    """根据bbox_format参数格式化边界框用于query"""
    # 格式化为 [x0, y0, x1, y1]字符串
    return f"[{','.join([str(int(coord)) for coord in bbox])}]"

def draw_result_image(image, detections):
    """绘制结果图像，显示框和验证结果"""
    if image is None:
        return None
        
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    try:
        font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
    except:
        font = ImageFont.load_default()
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        # 根据验证结果选择颜色
        color = (0, 255, 0) if det.get('qwen_verified', False) else (255, 0, 0)
        
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 添加标签
        label = f"{det.get('class_name', 'Region')} {'✓' if det.get('qwen_verified', False) else '✗'}"
        draw.text((x1 + 5, y1 + 5), label, fill=color, font=font)
    
    return vis_image

class QwenVLvLLM:
    """Qwen-VL vLLM推理类"""
    def __init__(self, 
                 checkpoint_path="/mnt/data/lyf/Qwen3-VL-32B-Instruct",
                 yolo_model_path=None):
        self.checkpoint_path = checkpoint_path
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        
        # 初始化vLLM模型
        self.llm = LLM(
            model=checkpoint_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
            enforce_eager=False,
            max_model_len=41960,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
            dtype="bfloat16",
        )
        
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=9280,
            top_k=-1,
            stop_token_ids=[],
        )
        
        # 初始化YOLOv8模型（可选）
        self.yolo_model = None
        if yolo_model_path:
            print(f"加载YOLOv8模型: {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path)
            self.yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print("YOLOv8模型加载成功!")
    
    def detect_with_yolov8(self, image_path: str):
        """
        使用YOLOv8进行目标检测
        
        Returns:
            Tuple[图像, 检测框列表]
        """
        if self.yolo_model is None:
            return None, "请先初始化YOLOv8模型"
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None, f"无法读取图像: {image_path}"
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # YOLOv8检测
        results = self.yolo_model(
            image_path,
            conf=0.25,  # 置信度阈值
            iou=0.45,   # IOU阈值
        )
        
        # 解析检测结果
        detections = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            if boxes is not None:
                for cls_id, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
                    # 获取框坐标和置信度
                    x1, y1, x2, y2 = xyxy
                    conf = float(conf)
                    cls_id = int(cls_id)
                    cls_name = self.yolo_model.names[cls_id]
                    
                    # 转换为整数坐标
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    # 过滤掉宽度和高度小于50的检测框
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width < 50 or box_height < 50:
                        print(f"过滤掉小框: {box_width}x{box_height}")
                        continue
                    
                    # 保存检测信息
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'qwen_verified': False,
                        'qwen_response': None,
                    }
                    detections.append(detection)
        
        # 转换为PIL图像用于显示
        pil_image = Image.fromarray(image_rgb)
        
        return pil_image, detections
    
    def analyze_yolo_detections(self, image_path, detections, custom_query=None):
        """使用Qwen-VL对YOLOv8检测框进行二次研判"""
        if not detections:
            return "", "未检测到任何目标", None
        
        try:
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            
            # 准备询问
            prompt_template = "你是一个yolo目标检测结果的研判专家，请根据图像上下文进行研判。" + query_class
            prompt_parts = []
            
            for i, det in enumerate(detections):
                # 裁剪矩形区域
                x0, y0, x1, y1 = det['bbox']
                # 归一化坐标到[0, 1000]范围
                x0_norm = int(x0 / image_width * 999)
                y0_norm = int(y0 / image_height * 999)
                x1_norm = int(x1 / image_width * 999)
                y1_norm = int(y1 / image_height * 999)

                # 添加提示词           
                coords_norm =[x0_norm, y0_norm, x1_norm, y1_norm]
                region_coords_str = _format_bbox_for_query(coords_norm)
        
                query = custom_query if custom_query else query_class
                prompt_parts.append(f"区域{i+1}-坐标：{region_coords_str}")

            maskimg = draw_visual_prompt(image, detections)
            prompt = f"{prompt_template}\n\n" + "\n".join(prompt_parts)
            #回复格式要求
            prompt += "\n请严格按照以下格式回复：\n区域1：yes, brief reason 或 区域1：no, brief reason\n区域2：yes, brief reason 或 区域2：no, brief reason\n..."
            
            # 准备模型输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": maskimg,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # 获取输入提示词
            input_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 调用模型
            inputs = [prepare_inputs_for_vllm(messages, self.processor)]
            outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            # 解析回答
            lines = generated_text.strip().split('\n')
            for i, det in enumerate(detections):
                region_key = f"区域{i+1}"
                region_found = False
                
                # 查找该区域的回答
                for line in lines:
                    if line.strip().startswith(region_key):
                        region_found = True
                        line_lower = line.lower()
                        if 'yes' in line_lower:
                            det['qwen_verified'] = True
                        elif 'no' in line_lower:
                            det['qwen_verified'] = False
                        det['qwen_response'] = line.strip()
                        break
                
                # 如果没找到该区域的回答
                if not region_found:
                    det['qwen_verified'] = False
            
            # 构建结果文本
            result_text = f"检测到 {len(detections)} 个目标：\n\n"
            for i, det in enumerate(detections):
                result_text += f"区域{i+1}:\n"
                if 'class_name' in det:
                    result_text += f"  YOLO类别: {det['class_name']}, 置信度: {det['confidence']:.2f}\n"
                result_text += f"  坐标: [{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]\n"
                result_text += f"  Qwen-VL验证: {'✓ 通过' if det['qwen_verified'] else '✗ 未通过'}\n"
                if det['qwen_response']:
                    result_text += f"  响应: {det['qwen_response']}\n"
                result_text += "\n"
            
            # 可视化结果
            # vis_image = draw_result_image(image, detections)
            
            return input_prompt, result_text, maskimg
            
        except Exception as e:
            error_msg = f"YOLOv8检测框分析过程中出现错误: {str(e)}"
            return "", error_msg, None
    
    def analyze_manual_boxes(self, annotation_data, custom_query=None):
        """使用Qwen-VL对手动绘制的框进行二次研判"""
        try:
            if not annotation_data or "image" not in annotation_data:
                return "", "请先上传图像并绘制区域", None
            
            image = annotation_data["image"]
            boxes_data = annotation_data.get("boxes", [])
            
            if not boxes_data:
                return "", "请在图像上绘制矩形框", image
            
           
            
            # 转换框数据格式
            detections = []
            for i, box in enumerate(boxes_data):
                if "bbox" in box:  # YOLO检测框格式
                    xmin, ymin, xmax, ymax = box["bbox"]
                else:  # 手动标注格式
                    xmin = box["xmin"]
                    ymin = box["ymin"]
                    xmax = box["xmax"] 
                    ymax = box["ymax"]
                
                detection = {
                    'bbox': [xmin, ymin, xmax, ymax],
                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                    'qwen_verified': False,
                    'qwen_response': None,
                }
                detections.append(detection)
            
            # 使用analyze_yolo_detections方法进行分析
            # 保存图像到临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                return self.analyze_yolo_detections(tmp_path, detections, custom_query)
            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
        except Exception as e:
            error_msg = f"手动框分析过程中出现错误: {str(e)}"
            return "", error_msg, None

# 创建Gradio界面
def create_gradio_interface(model_handler):
    with gr.Blocks(title="Qwen-VL + YOLOv8 目标检测与分析", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🐖 Qwen-VL + YOLOv8 目标检测与分析 🐂
        结合YOLOv8目标检测和Qwen-VL智能研判，实现自动化的目标识别与分析。
        """)
        
        with gr.Row():
            # 左侧：图像标注区域
            with gr.Column(scale=2):
                # 图像上传和标注组件（合二为一）
                annotator_component = image_annotator(
                    None,
                    label_list=["猪", "马", "牛", "羊", "其他"],
                    label_colors=additional_colors[:5],
                    box_thickness=3,
                    handle_size=8,
                    use_default_label=True,
                    image_type="pil",
                    boxes_alpha=0,
                    label="上传图像并标注区域",
                    height=500
                )
                
                # 研判提示词输入
                query_input = gr.Textbox(
                    label="研判提示词",
                    value=query_class,
                    placeholder="请输入对框内目标的研判提示词",
                    lines=2
                )
                
                # 按钮区域
                with gr.Row():
                    yolo_detect_btn = gr.Button("🚀 YOLOv8检测", variant="primary", size="lg")
                    manual_analyze_btn = gr.Button("✏️ 手动画框检测", variant="secondary", size="lg")
                    clear_btn = gr.Button("🗑️ 清除", variant="stop", size="sm")
            
            # 右侧：结果显示区域
            with gr.Column(scale=3):
                # 输入提示词显示
                input_prompt_display = gr.Textbox(
                    label="📝 输入提示词",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    show_copy_button=True
                )
                
                # 模型输出显示
                text_output = gr.Textbox(
                    label="🤖 模型输出",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                
                # 可视化结果
                image_output = gr.Image(label="🖼️ 结果可视化", height=500)
        
        def detect_with_yolo(image_annotation_data, custom_query):
            """YOLOv8检测函数"""
            if not image_annotation_data or "image" not in image_annotation_data:
                return "", "请先上传图像", None, image_annotation_data
            
            # 保存图像到临时文件
            image = image_annotation_data["image"]
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # 使用YOLOv8检测
                image, detections = model_handler.detect_with_yolov8(tmp_path)
                
                if isinstance(detections, str):  # 错误信息
                    return "", detections, None, image_annotation_data
                
                # 将检测结果转换为标注数据格式
                boxes_data = []
                for det in detections:
                    boxes_data.append({
                        "xmin": det['xmin'],
                        "ymin": det['ymin'],
                        "xmax": det['xmax'],
                        "ymax": det['ymax'],
                        "label": det['class_name']
                    })
                
                # 创建新的标注数据
                new_annotation_data = {
                    "image": image,
                    "boxes": boxes_data
                }
                
                # 使用Qwen-VL分析检测结果
                input_prompt, result_text, vis_image = model_handler.analyze_yolo_detections(
                    tmp_path, 
                    detections,
                    custom_query
                )
                
                return input_prompt, result_text, vis_image, new_annotation_data
            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        def analyze_manual_boxes(annotation_data, custom_query):
            """手动画框检测函数"""
            return model_handler.analyze_manual_boxes(annotation_data, custom_query)
        
        def clear_annotation():
            """清除标注"""
            return None
        
        # 事件绑定
        yolo_detect_btn.click(
            fn=detect_with_yolo,
            inputs=[annotator_component, query_input],
            outputs=[input_prompt_display, text_output, image_output, annotator_component]
        )
        
        manual_analyze_btn.click(
            fn=analyze_manual_boxes,
            inputs=[annotator_component, query_input],
            outputs=[input_prompt_display, text_output, image_output]
        )
        
        clear_btn.click(
            fn=clear_annotation,
            inputs=[],
            outputs=[annotator_component]
        )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 工作流程说明
            
            #### 方法一：🚀 YOLOv8自动检测
            1. **上传图像**：在左侧图像区域上传图片
            2. **设置提示词**：修改"研判提示词"（可选，默认为判断是否是猪马牛羊）
            3. **点击检测**：点击"YOLOv8检测"按钮，系统将：
               - 使用YOLOv8进行目标检测
               - 自动过滤掉小目标（宽高<50像素）
               - 使用Qwen-VL进行二次研判
               - 显示检测结果和分析
            
            #### 方法二：✏️ 手动画框检测
            1. **上传图像**：在左侧图像区域上传图片
            2. **绘制矩形框**：
               - 在图像上按住鼠标左键并拖动来绘制矩形框
               - 可以绘制多个矩形框
               - 绘制完成后可以拖动调整位置或大小
            3. **设置提示词**：修改"研判提示词"（可选）
            4. **点击分析**：点击"手动画框检测"按钮，系统将：
               - 使用Qwen-VL对绘制的框进行二次研判
               - 显示分析结果
            
            #### 按钮说明
            - **YOLOv8检测**：自动检测目标并分析
            - **手动画框检测**：分析手动绘制的框
            - **清除**：清除当前图像和标注
            
            #### 输出说明
            - **输入提示词**：显示发送给模型的完整提示词
            - **模型输出**：显示模型的原始输出文本  
            - **结果可视化**：显示带有区域框的可视化图像
            """)
    
    return demo

if __name__ == "__main__":
    # 初始化模型
    print("正在加载模型...")
    
    # 请根据您的实际路径修改
    yolo_model_path = "/mnt/disk/lyf/qwen/best.pt"  # YOLOv8模型路径
    qwen_model_path = "/mnt/disk/lyf/Qwen3-VL-4B-Instruct"  # Qwen-VL模型路径
    
    model_handler = QwenVLvLLM(
        checkpoint_path=qwen_model_path,
        yolo_model_path=yolo_model_path
    )
    print("模型加载完成！")
    
    # 创建并启动界面
    demo = create_gradio_interface(model_handler)
    demo.launch(
        server_name="0.0.0.0",
        server_port=9005,
        share=False
    )