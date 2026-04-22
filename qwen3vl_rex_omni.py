# -*- coding: utf-8 -*-
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import json
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import tempfile
import ast

# Qwen3-VL 相关导入
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Rex-Omni 相关导入
from rex_omni import RexOmniWrapper

class AutoDataGroundingAgent:
    """自动数据标注智能体"""
    
    def __init__(self, qwen_checkpoint_path="/mnt/data/lyf/Qwen3-VL-32B-Instruct", 
                 rex_model_path="/mnt/data/lyf/IDEA-Research/Rex-Omni"):
        
        # 初始化Qwen3-VL模型
        print("正在加载Qwen3-VL-32B模型...")
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_checkpoint_path)
        
        self.qwen_llm = LLM(
            model=qwen_checkpoint_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.6,
            enforce_eager=False,
            max_model_len=41960,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
            dtype="bfloat16",
        )
        
        self.qwen_sampling_params = SamplingParams(
            temperature=0,
            max_tokens=9280,
            top_k=-1,
            stop_token_ids=[],
        )
        print("Qwen3-VL-32B模型加载完成！")
        
        # 初始化Rex-Omni模型
        print("正在加载Rex-Omni模型...")
        self.rex_model = RexOmniWrapper(
            model_path=rex_model_path,
            backend="vllm",
            max_tokens=40960,
            gpu_memory_utilization=0.3,
            temperature=0.0
        )
        print("Rex-Omni模型加载完成！")
    
    def prepare_qwen_inputs(self, messages):
        """准备Qwen-VL输入"""
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.qwen_processor.image_processor.patch_size,
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
    
    def detect_categories_with_qwen(self, image):
        """使用Qwen3-VL检测图像中的视觉类别"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": "请列出这张图像中出现的所有视觉目标类别（如建筑，轿车，压路机，电动车，耕地等），用逗号分隔。只返回类别名称，不要有其他描述。"},
                    ],
                }
            ]
            
            inputs = [self.prepare_qwen_inputs(messages)]
            outputs = self.qwen_llm.generate(inputs, sampling_params=self.qwen_sampling_params)
            category_text = outputs[0].outputs[0].text.strip()
            
            # 解析类别文本
            categories = [cat.strip() for cat in category_text.split(',') if cat.strip()]
            return categories, category_text
            
        except Exception as e:
            return [], f"类别检测失败: {str(e)}"
    
    def detect_objects_with_rex_omni(self, image, categories):
        """使用Rex-Omni检测指定类别的对象"""
        try:
            # 执行Rex-Omni推理
            results = self.rex_model.inference(
                images=image,
                task="detection",
                categories=categories
            )
            
            result = results[0]
            if not result.get("success", False):
                raise RuntimeError(f"Rex-Omni推理失败: {result.get('error', '未知错误')}")
            
            predictions = result["extracted_predictions"]
            
            # 转换为标准格式的框数据
            boxes_data = []
            for category, detections in predictions.items():
                for detection in detections:
                    if detection["type"] == "box":
                        coords = detection["coords"]
                        boxes_data.append({
                            "xmin": coords[0],
                            "ymin": coords[1],
                            "xmax": coords[2],
                            "ymax": coords[3],
                            "label": category,
                            "confidence": detection.get("confidence", 0.0)
                        })
            
            return boxes_data, predictions
            
        except Exception as e:
            return [], f"对象检测失败: {str(e)}"
    
    def relabel_boxes_with_qwen(self, image, boxes_data):
        """使用Qwen3-VL重新标定矩形框的类别和描述"""
        try:
            if not boxes_data:
                return [], "没有检测到任何对象"
            
            width, height = image.size
            
            # 格式化框数据用于提示词
            formatted_boxes = []
            for i, box in enumerate(boxes_data):
                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                label =  box["label"]
                # 转换为归一化坐标 [0, 1000]
                norm_x1 = int(xmin * 1000 / width)
                norm_y1 = int(ymin * 1000 / height)
                norm_x2 = int(xmax * 1000 / width)
                norm_y2 = int(ymax * 1000 / height)
                
                formatted_boxes.append({
                    "bbox_2d": [norm_x1, norm_y1, norm_x2, norm_y2],
                    "region_id": str(i + 1)
                })
            
            # 构建重新标定提示词
            relabel_prompt = """请对以下每个区域进行详细描述，输出格式为：
            region_id: label|brief instance description

            要求：
            1. label使用中文名词
            2. description用中文简要描述该物体的特征、状态等
            3. 每个区域单独一行

            例如：
            1: 人|一个穿着红色衣服的年轻人在走路
            2: 车辆|一辆白色的轿车停在路边

            请开始描述："""
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": relabel_prompt},
                        {
                            "type": "text", 
                            "text": f"以下是需要关注的区域坐标：{json.dumps(formatted_boxes, ensure_ascii=False)}"
                        }
                    ],
                }
            ]
            
            inputs = [self.prepare_qwen_inputs(messages)]
            outputs = self.qwen_llm.generate(inputs, sampling_params=self.qwen_sampling_params)
            relabel_result = outputs[0].outputs[0].text.strip()
            
            # 解析重新标定结果
            relabeled_boxes = self.parse_relabel_result(relabel_result, boxes_data)
            
            return relabeled_boxes, relabel_result
            
        except Exception as e:
            return [], f"重新标定失败: {str(e)}"
    
    def parse_relabel_result(self, relabel_text, original_boxes):
        """解析重新标定结果"""
        relabeled_boxes = []
        lines = [line for line in relabel_text.splitlines() if line.strip()]
        print(lines)
        print(original_boxes)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # 解析格式：region_id: label|description
            parts = line.split(':', 1)
            if len(parts) < 2:
                continue
            
            region_info = parts[1].strip()
            if '|' in region_info:
                label_part, description = region_info.split('|', 1)
                label = label_part.strip()
                description = description.strip()
            else:
                label = region_info.strip()
                description = ""
            
            # 获取对应的原始框数据
            if i < len(original_boxes):
                box_data = original_boxes[i].copy()
                box_data["relabel"] = label
                box_data["description"] = description
                relabeled_boxes.append(box_data)
        
        return relabeled_boxes
    
    def draw_boxes_on_image(self, image, boxes_data, title="检测结果"):
        """在图像上绘制矩形框和标签"""
        if image is None or not boxes_data:
            return image
            
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        try:
            font = ImageFont.truetype("./wqy-microhei.ttc", size=20)
        except:
            font = ImageFont.load_default()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255), 
                 (255, 165, 0), (255, 192, 203), (128, 0, 128)]
        
        for i, box in enumerate(boxes_data):
            color = colors[i % len(colors)]
            
            xmin = box["xmin"]
            ymin = box["ymin"] 
            xmax = box["xmax"]
            ymax = box["ymax"]
            
            # 绘制矩形框
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            
            # 准备标签文本
            if "relabel" in box and "description" in box:
                label_text = f"{box['relabel']}: {box['description']}"
            else:
                label_text = box.get("label", f"Region {i+1}")
            
            # 添加标签背景
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.rectangle([xmin, ymin - text_height - 5, xmin + text_width + 10, ymin], 
                         fill=color)
            
            # 添加标签文本
            draw.text((xmin + 5, ymin - text_height - 2), label_text, fill=(255, 255, 255), font=font)
        
        # 添加标题
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text((width - title_width - 10, 10), title, fill=(0, 0, 0), font=font)
        
        return img

def create_auto_grounding_interface(agent):
    """创建自动数据标注界面"""
    
    with gr.Blocks(title="自动数据标注智能体", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 自动数据标注智能体
        基于Qwen3-VL-32B和Rex-Omni的智能数据标注系统
        
        **工作流程**:
        1. **Qwen3-VL** → 识别图像中的视觉类别
        2. **Rex-Omni** → 检测指定类别的对象边界框  
        3. **Qwen3-VL** → 重新标定边界框的类别和描述
        """)
        
        with gr.Row():
            with gr.Column():
                # 图像上传
                input_image = gr.Image(
                    label="上传图像",
                    type="pil",
                    height=400
                )
                
                # 处理按钮
                process_btn = gr.Button("🚀 开始自动标注", variant="primary", size="lg")
                
                # 手动类别输入（可选）
                with gr.Accordion("手动指定类别（可选）", open=False):
                    manual_categories = gr.Textbox(
                        label="手动输入类别（逗号分隔）",
                        placeholder="例如: person,car,dog,cat",
                        lines=2
                    )
                    use_manual_categories = gr.Checkbox(label="使用手动输入的类别", value=False)
            
            with gr.Column():
                # 第一步：Qwen3-VL类别检测结果
                with gr.Tab("1. 类别识别"):
                    qwen_categories_output = gr.Textbox(
                        label="Qwen3-VL识别的类别",
                        lines=3,
                        interactive=False
                    )
                
                # 第二步：Rex-Omni检测结果
                with gr.Tab("2. 对象检测"):
                    rex_detection_output = gr.Textbox(
                        label="Rex-Omni检测结果",
                        lines=5,
                        interactive=False
                    )
                    rex_visualization = gr.Image(
                        label="Rex-Omni检测可视化",
                        height=400
                    )
                
                # 第三步：Qwen3-VL重新标定结果
                with gr.Tab("3. 重新标定"):
                    qwen_relabel_output = gr.Textbox(
                        label="Qwen3-VL重新标定结果",
                        lines=8,
                        interactive=False
                    )
                    qwen_visualization = gr.Image(
                        label="最终标注结果",
                        height=400
                    )
        
        def process_auto_grounding(image, manual_categories_text, use_manual):
            """处理自动数据标注流程"""
            if image is None:
                return "请先上传图像", "", "", None, "", None
            
            try:
                # 第一步：使用Qwen3-VL检测类别
                if use_manual and manual_categories_text.strip():
                    categories = [cat.strip() for cat in manual_categories_text.split(',') if cat.strip()]
                    qwen_categories_output = f"使用手动输入类别: {', '.join(categories)}"
                else:
                    categories, qwen_categories_output = agent.detect_categories_with_qwen(image)
                    if not categories:
                        return "未检测到任何类别", "", "", None, "", None
                
                # 第二步：使用Rex-Omni检测对象
                boxes_data, rex_raw_output = agent.detect_objects_with_rex_omni(image, categories)
                rex_detection_text = f"检测到 {len(boxes_data)} 个对象:\n"
                for i, box in enumerate(boxes_data):
                    rex_detection_text += f"{i+1}. {box['label']} - 置信度: {box.get('confidence', 0):.3f}\n"
                
                # 绘制Rex-Omni检测结果
                rex_image = agent.draw_boxes_on_image(image, boxes_data, "Rex-Omni检测结果")
                
                # 第三步：使用Qwen3-VL重新标定
                relabeled_boxes, qwen_relabel_text = agent.relabel_boxes_with_qwen(image, boxes_data)
                
                # 绘制最终标注结果
                final_image = agent.draw_boxes_on_image(image, relabeled_boxes, "最终标注结果")
                
                return (qwen_categories_output, rex_detection_text, rex_raw_output, 
                       rex_image, qwen_relabel_text, final_image)
                
            except Exception as e:
                error_msg = f"处理过程中出现错误: {str(e)}"
                return error_msg, "", "", None, "", None
        
        # 绑定事件
        process_btn.click(
            fn=process_auto_grounding,
            inputs=[input_image, manual_categories, use_manual_categories],
            outputs=[
                qwen_categories_output,
                rex_detection_output,
                rex_detection_output,  # 重复使用，显示原始输出
                rex_visualization,
                qwen_relabel_output,
                qwen_visualization
            ]
        )
        
        # 使用说明
        gr.Markdown("""
        ## 使用说明
        
        ### 自动模式（推荐）
        1. 上传图像
        2. 点击"开始自动标注"按钮
        3. 系统会自动执行以下流程：
           - **步骤1**: Qwen3-VL识别图像中的所有视觉类别
           - **步骤2**: Rex-Omni检测这些类别的对象边界框
           - **步骤3**: Qwen3-VL重新标定每个边界框的类别和描述
        
        ### 手动模式
        1. 上传图像
        2. 展开"手动指定类别"面板
        3. 输入您想要检测的类别（英文，逗号分隔）
        4. 勾选"使用手动输入的类别"
        5. 点击"开始自动标注"按钮
        
        ### 输出说明
        - **类别识别**: 显示Qwen3-VL识别的图像类别列表
        - **对象检测**: 显示Rex-Omni的检测结果和可视化
        - **重新标定**: 显示Qwen3-VL的重新标定结果和最终可视化
        
        ### 标注格式
        最终标注采用格式：`label|brief instance description`
        - `label`: 英文类别名称
        - `description`: 中文实例描述
        """)
    
    return demo

if __name__ == "__main__":
    # 初始化智能体
    print("正在初始化自动数据标注智能体...")
    agent = AutoDataGroundingAgent()
    print("智能体初始化完成！")
    
    # 创建并启动界面
    demo = create_auto_grounding_interface(agent)
    demo.launch(
        server_name="0.0.0.0",
        server_port=9006,
        share=True
    )
