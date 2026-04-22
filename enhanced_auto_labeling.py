# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import torch
import json
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import tempfile
import ast
import numpy as np
from typing import List, Dict, Tuple, Optional

# Qwen3-VL 相关导入
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Rex-Omni 相关导入
from rex_omni import RexOmniWrapper

class EnhancedAutoDataGroundingAgent:
    """增强版自动数据标注智能体 - 支持补标漏检目标"""
    
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
    
    def get_image_categories(self, image, manual_categories: Optional[str] = None) -> Tuple[List[str], str]:
        """获取图像类别：支持手动输入或自动检测"""
        if manual_categories and manual_categories.strip():
            categories = [cat.strip() for cat in manual_categories.split(',') if cat.strip()]
            return categories, f"{', '.join(categories)}"
        
        # 自动检测类别
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
            return categories, f"自动检测类别: {', '.join(categories)}"
            
        except Exception as e:
            return [], f"类别检测失败: {str(e)}"
    
    def detect_objects_with_rex_omni(self, image, categories: List[str]) -> Tuple[List[Dict], str]:
        """使用Rex-Omni检测指定类别的对象"""
        try:
            if not categories:
                return [], "没有指定检测类别"
            
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
                            "xmin": float(coords[0]),
                            "ymin": float(coords[1]),
                            "xmax": float(coords[2]),
                            "ymax": float(coords[3]),
                            "label": category,
                            "confidence": float(detection.get("confidence", 0.0)),
                            "source": "rex_omni"
                        })
            
            # 生成检测结果文本
            detection_text = f"Rex-Omni检测到 {len(boxes_data)} 个对象:\n"
            for i, box in enumerate(boxes_data):
                detection_text += f"{i+1}. {box['label']} - 置信度: {box['confidence']:.3f}\n"
            
            return boxes_data, detection_text
            
        except Exception as e:
            return [], f"对象检测失败: {str(e)}"
    
    def relabel_boxes_with_qwen(self, image, boxes_data: List[Dict]) -> Tuple[List[Dict], str]:
        """使用Qwen3-VL重新标定矩形框的类别和描述"""
        try:
            if not boxes_data:
                return [], "没有检测到任何对象"
            
            width, height = image.size
            
            # 格式化框数据用于提示词
            formatted_boxes = []
            for i, box in enumerate(boxes_data):
                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                
                # 转换为归一化坐标 [0, 1000]
                norm_x1 = int(xmin * 1000 / width)
                norm_y1 = int(ymin * 1000 / height)
                norm_x2 = int(xmax * 1000 / width)
                norm_y2 = int(ymax * 1000 / height)
                
                formatted_boxes.append({
                    "bbox_2d": [norm_x1, norm_y1, norm_x2, norm_y2],
                    "region_id": str(i + 1),
                    "original_label": box.get("label", "")
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
            relabeled_boxes = self._parse_relabel_result(relabel_result, boxes_data)
            
            # 生成重新标定结果文本
            relabel_text = f"Qwen3-VL重新标定结果 ({len(relabeled_boxes)} 个对象):\n"
            for i, box in enumerate(relabeled_boxes):
                original_label = box.get("original_label", "")
                new_label = box.get("relabel", "")
                description = box.get("description", "")
                relabel_text += f"{i+1}. 原始: {original_label} -> 新: {new_label} | {description}\n"
            
            return relabeled_boxes, relabel_text
            
        except Exception as e:
            return [], f"重新标定失败: {str(e)}"
    
    def detect_missing_objects_with_qwen(self, image, existing_boxes: List[Dict], categories: List[str]) -> Tuple[List[Dict], str]:
        """使用Qwen3-VL检测遗漏的目标"""
        try:
            # 将已有框绘制到图像上
            image_with_boxes = self._draw_boxes_on_image_internal(image, existing_boxes, draw_labels=False)
            
            # 构建检测遗漏目标的提示词
            missing_prompt = """请仔细观察这张图像，图像中已经有了一些矩形框标注，请找出图中明显存在、但被遗漏的目标物体，仅检测以下类别：""" + ", ".join(categories) + """。以JSON格式报告他们的边界框坐标,其中label以类似：label/brief described instance这种形式给出"""
            print(missing_prompt)
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_with_boxes,
                        },
                        {"type": "text", "text": missing_prompt}
                    ],
                }
            ]
            
            inputs = [self.prepare_qwen_inputs(messages)]
            outputs = self.qwen_llm.generate(inputs, sampling_params=self.qwen_sampling_params)
            missing_result = outputs[0].outputs[0].text.strip()
            
            # 解析JSON输出
            missing_boxes = self._parse_missing_boxes_json(missing_result, image.size)
            
            # 生成补标结果文本
            missing_text = f"Qwen3-VL补标检测到 {len(missing_boxes)} 个遗漏对象:\n"
            for i, box in enumerate(missing_boxes):
                missing_text += f"{i+1}. {box.get('label', '未知')}: {box.get('description', '')}\n"
            
            return missing_boxes, missing_text
            
        except Exception as e:
            return [], f"遗漏目标检测失败: {str(e)}"
    
    def _parse_relabel_result(self, relabel_text: str, original_boxes: List[Dict]) -> List[Dict]:
        """解析重新标定结果"""
        relabeled_boxes = []
        lines = [line for line in relabel_text.splitlines() if line.strip()]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # 解析格式：region_id: label|description
            parts = line.split(':', 1)
            if len(parts) < 2:
                continue
            
            region_id = parts[0].strip()
            region_info = parts[1].strip()
            
            if '|' in region_info:
                label_part, description = region_info.split('|', 1)
                label = label_part.strip()
                description = description.strip()
            else:
                label = region_info.strip()
                description = ""
            
            # 获取对应的原始框数据
            try:
                idx = int(region_id) - 1
                if 0 <= idx < len(original_boxes):
                    box_data = original_boxes[idx].copy()
                    box_data["relabel"] = label
                    box_data["description"] = description
                    box_data["source"] = "rex_omni_relabeled"
                    relabeled_boxes.append(box_data)
            except ValueError:
                continue
        
        return relabeled_boxes
    
    def _parse_missing_boxes_json(self, json_text: str, image_size: Tuple[int, int]) -> List[Dict]:
        """解析遗漏目标的JSON输出"""
        try:
            # 清理markdown标记
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            
            # 解析JSON
            data = json.loads(json_text)
            
            width, height = image_size
            missing_boxes = []
            
            for item in data:
                if "bbox_2d" in item:
                    # 归一化坐标转换回像素坐标
                    norm_coords = item["bbox_2d"]
                    if len(norm_coords) == 4:
                        x1, y1, x2, y2 = norm_coords
                        
                        # 确保坐标顺序正确
                        x1, x2 = sorted([x1, x2])
                        y1, y2 = sorted([y1, y2])
                        
                        # 转换为像素坐标
                        pixel_x1 = int(x1 * width / 1000)
                        pixel_y1 = int(y1 * height / 1000)
                        pixel_x2 = int(x2 * width / 1000)
                        pixel_y2 = int(y2 * height / 1000)
                        
                        missing_boxes.append({
                            "xmin": pixel_x1,
                            "ymin": pixel_y1,
                            "xmax": pixel_x2,
                            "ymax": pixel_y2,
                            "label": item.get("label", "未知"),
                            "description": item.get("description", ""),
                            "source": "qwen_missing"
                        })
            
            return missing_boxes
            
        except Exception as e:
            print(f"解析遗漏目标JSON失败: {e}")
            return []
    
    def _draw_boxes_on_image_internal(self, image: Image.Image, boxes_data: List[Dict], 
                                    title: str = "", draw_labels: bool = True) -> Image.Image:
        """内部方法：在图像上绘制矩形框"""
        if image is None or not boxes_data:
            return image
            
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        try:
            font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
        except:
            font = ImageFont.load_default()
        
        # 不同来源的框使用不同颜色
        source_colors = {
            "rex_omni": (255, 0, 0, 255),      # 红色 - Rex-Omni原始检测
            "rex_omni_relabeled": (0, 255, 0, 255),  # 绿色 - 重新标定后的
            "qwen_missing": (0, 0, 255, 255),   # 蓝色 - 补标的遗漏目标
        }
        
        for i, box in enumerate(boxes_data):
            source = box.get("source", "unknown")
            color = source_colors.get(source, (255, 165, 0, 255))  # 默认橙色
            
            xmin = box["xmin"]
            ymin = box["ymin"] 
            xmax = box["xmax"]
            ymax = box["ymax"]
            
            # 绘制矩形框
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
            
            if draw_labels:
                # 准备标签文本
                label_text = ""
                if "relabel" in box and box["relabel"]:
                    label_text = f"{box['relabel']}"
                else:
                    label_text = box.get("label", f"Region {i+1}")
                
                # 添加简短描述（如果存在）
                # if "description" in box and box["description"]:
                #     desc = box["description"][:20] + "..." if len(box["description"]) > 20 else box["description"]
                #     label_text += f": {desc}"
                
                # 添加标签背景
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # 背景框
                draw.rectangle([xmin, ymin - text_height - 4, xmin + text_width + 8, ymin], 
                             fill=color)
                
                # 标签文本
                draw.text((xmin + 4, ymin - text_height - 2), label_text, fill=(255, 255, 255), font=font)
        
        # 添加标题
        if title:
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            draw.text((width - title_width - 10, 10), title, fill=(0, 0, 0), font=font)
        
        return img
    
    def draw_rex_omni_result(self, image: Image.Image, boxes_data: List[Dict]) -> Image.Image:
        """绘制Rex-Omni检测结果图"""
        rex_boxes = [box for box in boxes_data if box.get("source") == "rex_omni"]
        return self._draw_boxes_on_image_internal(image, rex_boxes, "Rex-Omni检测结果", draw_labels=True)
    
    def draw_qwen_relabel_result(self, image: Image.Image, boxes_data: List[Dict]) -> Image.Image:
        """绘制Qwen3-VL校正结果图"""
        relabeled_boxes = [box for box in boxes_data if box.get("source") == "rex_omni_relabeled"]
        return self._draw_boxes_on_image_internal(image, relabeled_boxes, "Qwen3-VL校正结果", draw_labels=True)
    
    def draw_qwen_missing_result(self, image: Image.Image, boxes_data: List[Dict]) -> Image.Image:
        """绘制Qwen3-VL补标结果图"""
        missing_boxes = [box for box in boxes_data if box.get("source") == "qwen_missing"]
        return self._draw_boxes_on_image_internal(image, missing_boxes, "Qwen3-VL补标结果", draw_labels=True)
    
    def draw_final_result(self, image: Image.Image, boxes_data: List[Dict]) -> Image.Image:
        """绘制最终完整结果图"""
        return self._draw_boxes_on_image_internal(image, boxes_data, "最终完整检测结果", draw_labels=True)

def create_enhanced_grounding_interface(agent):
    """创建增强版自动数据标注界面"""
    
    with gr.Blocks(title="增强版自动数据标注系统", theme=gr.themes.Soft(), css="""
        .results-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }
        .result-image {
            flex: 1;
            min-width: 300px;
            max-width: 450px;
        }
        .text-output {
            font-family: monospace;
            font-size: 12px;
        }
    """) as demo:
        gr.Markdown("""
        # 🚀 增强版自动数据标注系统
        基于Rex-Omni和Qwen3-VL的智能数据标注系统，支持标签校正和漏检目标补标
        
        **完整工作流程**:
        1. **类别输入** → 手动指定或自动检测图像中的视觉类别
        2. **Rex-Omni检测** → 检测指定类别的对象边界框
        3. **Qwen3-VL校正** → 重新标定边界框的类别和描述
        4. **Qwen3-VL补标** → 检测遗漏的目标并补充标注
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 图像上传
                input_image = gr.Image(
                    label="上传图像",
                    type="pil",
                    height=350
                )
                
                # 类别输入
                with gr.Accordion("🔧 类别设置", open=True):
                    manual_categories = gr.Textbox(
                        label="手动输入检测类别（可选，英文逗号分隔）",
                        placeholder="例如: person,car,dog,cat,building,tree",
                        lines=2,
                        value=""
                    )
                    gr.Markdown("*留空则自动检测图像中的类别*")
                
                # 处理按钮
                process_btn = gr.Button("🚀 开始完整标注流程", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                # 文本输出区域
                with gr.Accordion("📋 处理日志", open=True):
                    log_output = gr.Textbox(
                        label="处理过程日志",
                        lines=10,
                        interactive=False,
                        elem_classes="text-output"
                    )
                
                # 结果图像展示
                gr.Markdown("### 📊 可视化结果")
                
                with gr.Row():
                    rex_image = gr.Image(
                        label="Rex-Omni检测结果",
                        height=300,
                        elem_classes="result-image"
                    )
                    relabel_image = gr.Image(
                        label="Qwen3-VL校正结果",
                        height=300,
                        elem_classes="result-image"
                    )
                
                with gr.Row():
                    missing_image = gr.Image(
                        label="Qwen3-VL补标结果",
                        height=300,
                        elem_classes="result-image"
                    )
                    final_image = gr.Image(
                        label="最终完整结果",
                        height=300,
                        elem_classes="result-image"
                    )
        
        def process_enhanced_grounding(image, categories_text):
            """处理增强版数据标注流程"""
            log_messages = []
            
            if image is None:
                return "请先上传图像", None, None, None, None
            
            try:
                # 第一步：获取类别
                log_messages.append("=== 步骤1: 获取检测类别 ===")
                categories, categories_info = agent.get_image_categories(image, categories_text)
                log_messages.append(categories_info)
                log_messages.append(f"检测类别: {', '.join(categories) if categories else '无'}")
                
                if not categories:
                    return "\n".join(log_messages), None, None, None, None
                
                # 第二步：Rex-Omni检测
                log_messages.append("\n=== 步骤2: Rex-Omni对象检测 ===")
                rex_boxes, rex_text = agent.detect_objects_with_rex_omni(image, categories)
                log_messages.append(rex_text)
                
                # 第三步：Qwen3-VL校正
                log_messages.append("\n=== 步骤3: Qwen3-VL标签校正 ===")
                relabeled_boxes, relabel_text = agent.relabel_boxes_with_qwen(image, rex_boxes)
                log_messages.append(relabel_text)
                
                # 第四步：Qwen3-VL补标遗漏目标
                log_messages.append("\n=== 步骤4: Qwen3-VL补标遗漏目标 ===")
                missing_boxes, missing_text = agent.detect_missing_objects_with_qwen(image, relabeled_boxes, categories)
                log_messages.append(missing_text)
                
                # 合并所有框
                all_boxes = []
                all_boxes.extend([box for box in rex_boxes if box.get("source") == "rex_omni"])
                all_boxes.extend(relabeled_boxes)
                all_boxes.extend(missing_boxes)
                
                # 生成结果统计
                log_messages.append("\n=== 最终统计 ===")
                rex_count = len([b for b in all_boxes if b.get("source") == "rex_omni"])
                relabel_count = len([b for b in all_boxes if b.get("source") == "rex_omni_relabeled"])
                missing_count = len([b for b in all_boxes if b.get("source") == "qwen_missing"])
                log_messages.append(f"Rex-Omni检测: {rex_count} 个对象")
                log_messages.append(f"Qwen3-VL校正: {relabel_count} 个对象")
                log_messages.append(f"Qwen3-VL补标: {missing_count} 个对象")
                log_messages.append(f"总计: {len(all_boxes)} 个对象")
                
                # 生成可视化图像
                rex_result = agent.draw_rex_omni_result(image, rex_boxes)
                relabel_result = agent.draw_qwen_relabel_result(image, relabeled_boxes)
                missing_result = agent.draw_qwen_missing_result(image, missing_boxes)
                final_result = agent.draw_final_result(image, all_boxes)
                
                return "\n".join(log_messages), rex_result, relabel_result, missing_result, final_result
                
            except Exception as e:
                log_messages.append(f"\n❌ 处理过程中出现错误: {str(e)}")
                return "\n".join(log_messages), None, None, None, None
        
        # 绑定事件
        process_btn.click(
            fn=process_enhanced_grounding,
            inputs=[input_image, manual_categories],
            outputs=[log_output, rex_image, relabel_image, missing_image, final_image]
        )
        
        # 使用说明
        gr.Markdown("""
        ## 📖 使用说明
        
        ### 基本流程
        1. **上传图像**: 选择需要标注的图像文件
        2. **设置类别** (可选):
           - 手动输入: 输入您想要检测的类别，英文逗号分隔（如 `person,car,dog`）
           - 自动检测: 留空文本框，系统会自动识别图像中的类别
        3. **开始标注**: 点击"开始完整标注流程"按钮
        
        ### 四步标注流程
        1. **类别获取**: 确定要检测的对象类别
        2. **Rex-Omni检测**: 基于指定类别进行对象检测
        3. **Qwen3-VL校正**: 对检测结果进行精细化标签校正
        4. **Qwen3-VL补标**: 检测并标注遗漏的目标对象
        
        ### 结果说明
        - **Rex-Omni检测结果**: 显示Rex-Omni模型的原始检测结果（红色框）
        - **Qwen3-VL校正结果**: 显示经过Qwen3-VL标签校正后的结果（绿色框）
        - **Qwen3-VL补标结果**: 显示Qwen3-VL检测到的遗漏目标（蓝色框）
        - **最终完整结果**: 显示所有检测框的合并结果
        
        ### 颜色标识
        - 🔴 **红色**: Rex-Omni原始检测框
        - 🟢 **绿色**: Qwen3-VL校正后的框
        - 🔵 **蓝色**: Qwen3-VL补标的遗漏目标框
        
        ### 注意事项
        1. 对于复杂图像，建议手动指定关键检测类别以提高准确性
        2. 补标阶段会检测图像中明显存在但被漏标的目标
        3. 处理时间取决于图像复杂度和模型加载状态
        4. 确保有足够的GPU内存运行两个大模型
        """)
    
    return demo

if __name__ == "__main__":
    # 初始化智能体
    print("正在初始化增强版自动数据标注智能体...")
    agent = EnhancedAutoDataGroundingAgent()
    print("智能体初始化完成！")
    
    # 创建并启动界面
    demo = create_enhanced_grounding_interface(agent)
    demo.launch(
        server_name="0.0.0.0",
        server_port=9009,
        share=True,
        debug=True
    )