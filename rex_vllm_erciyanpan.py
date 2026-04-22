import os
from platform import processor
from pyexpat.errors import messages
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional
import re
import time
from tqdm import tqdm
import math
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
# 直接导入YOLOv8，去掉try-except
from ultralytics import YOLO
YOLO_AVAILABLE = True

query_class = "这个矩形框的类别是动物吗？"  
class LabelmeAnnotation:
    """Labelme格式的标注数据结构"""
    
    def __init__(self, image_path: str, image_shape: Tuple[int, int, int]):
        """
        初始化Labelme标注
        
        Args:
            image_path: 图像路径
            image_shape: 图像形状 (height, width, channels)
        """
        self.version = "5.2.0"
        self.flags = {}
        self.shapes = []  # 存储所有形状标注
        self.imagePath = Path(image_path).name
        self.imageHeight = image_shape[0]
        self.imageWidth = image_shape[1]
        
    def add_shape(self, 
                  label: str, 
                  points: List[List[float]], 
                  shape_type: str = "rectangle",
                  attributes: Optional[Dict] = None,
                  group_id: Optional[int] = None):
        """
        添加一个形状标注
        
        Args:
            label: 类别标签
            points: 点坐标列表，对于矩形是[[x1, y1], [x2, y2]]
            shape_type: 形状类型 ("rectangle", "polygon", etc.)
            attributes: 附加属性
            group_id: 分组ID
        """
        shape = {
            "label": label,
            "points": points,
            "group_id": group_id,
            "shape_type": shape_type,
            "flags": {}
        }
        
        if attributes:
            shape["attributes"] = attributes
            
        self.shapes.append(shape)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "version": self.version,
            "flags": self.flags,
            "shapes": self.shapes,
            "imagePath": self.imagePath,
            "imageData": None,  # 不保存图像数据，只保存路径
            "imageHeight": self.imageHeight,
            "imageWidth": self.imageWidth
        }
    
    def save(self, output_path: str):
        """保存为JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

class YOLOv8QwenVLAnalyzer:
    """YOLOv8检测 + Qwen-VL二次研判分析器"""
    
    def __init__(self,
                 yolo_model_path: str = 'yolov8n.pt',
                 qwen_model_name: str = 'Qwen/Qwen2-VL-2B-Instruct',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 batch_size: int = 5,  # 每批询问的矩形框数量
                 max_retries: int = 3):
        """
        初始化分析器
        
        Args:
            yolo_model_path: YOLOv8模型路径
            qwen_model_name: Qwen-VL模型名称
            confidence_threshold: 检测置信度阈值
            iou_threshold: IOU阈值
            batch_size: 每次询问的矩形框数量
            max_retries: 最大重试次数
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # 初始化YOLOv8模型
        if not YOLO_AVAILABLE:
            raise ImportError("请安装ultralytics: pip install ultralytics")
        
        print(f"加载YOLOv8模型: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print("YOLOv8模型加载成功!")
        
        # 初始化Qwen-VL模型
        print(f"加载Qwen-VL模型: {qwen_model_name}")
        self.processor = AutoProcessor.from_pretrained(qwen_model_name)
        self.llm = LLM(
            model=qwen_model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
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
        print("Qwen-VL模型加载成功!")
    

    def prepare_inputs_for_vllm(self, messages, processor):
        """准备vLLM输入"""
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # qwen_vl_utils 0.0.14+ required
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


    def detect_with_yolov8(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        使用YOLOv8进行目标检测
        
        Returns:
            Tuple[原始图像, 检测框列表]
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # YOLOv8检测
        results = self.yolo_model(
            image_rgb,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # 解析检测结果
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # 获取框坐标和置信度
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.yolo_model.names[cls_id]
                    
                    # 转换为整数坐标
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    # 计算框的宽度和高度，并过滤掉宽度和高度小于50的检测框
                    box_width = x2 - x1
                    box_height = y2 - y1
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    # 过滤掉宽度和高度小于50的检测框
                    if box_width < 50 or box_height < 50:
                        continue
                    
                    # 保存检测信息
                    detection = {
                        'id': i,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'qwen_verified': False,  # Qwen-VL是否验证通过
                        'qwen_final_class': cls_name,  # 最终类别（可能被Qwen-VL修正）
                        'qwen_response': None,  # Qwen-VL的原始响应
                        'qwen_certainty': None,  # Qwen-VL的确定度
                    }
                    detections.append(detection)
        
        return image, detections
    
    def _format_bbox_for_query(self, bbox):
        """根据bbox_format参数格式化边界框用于query"""
        # 格式化为 <x0><y0><x1><y1>
        return ''.join([f'<{int(coord)}>' for coord in bbox])
    
    def batch_analyze_with_qwen_vl(self, 
                                   image_path: str, 
                                   detections: List[Dict]) -> List[Dict]:
        """
        批量使用Qwen-VL对检测框进行二次研判
        
        Args:
            image: 原始图像
            detections: 检测框列表
            
        Returns:
            更新后的检测框列表
        """
        if self.llm is None:
            print("Qwen-VL模型未加载，跳过二次研判")
            return detections
        
        if not detections:
            return detections
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        
        # 将检测框分批
        num_batches = math.ceil(len(detections) / self.batch_size)
        
        print(f"开始Qwen-VL二次研判，共{len(detections)}个检测框，分{num_batches}批处理")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(detections))
            batch_detections = detections[start_idx:end_idx]
            
            print(f"处理批次 {batch_idx + 1}/{num_batches}: 框 {start_idx}-{end_idx-1}")
            
            # 准备批次询问
            prompt_template = ["根据图片内容，回答以下关于特定区域的问题："]
           
            prompt_parts = []
            for i, det in enumerate(batch_detections):
                # 裁剪矩形区域
                x0, y0, x1, y1 = det['bbox']
                # 归一化坐标到[0, 1000]范围
                x0_norm = int(x0 / image_width * 1000)
                y0_norm = int(y0 / image_height * 1000)
                x1_norm = int(x1 / image_width * 1000)
                y1_norm = int(y1 / image_height * 1000)

                # 添加提示词           
                coords_norm =[x0_norm, y0_norm, x1_norm, y1_norm]
                region_coords_str = self._format_bbox_for_query(coords_norm)
                yolo_class = det['class_name']
                prompt_parts.append(f"区域{i+1}（坐标：{region_coords_str}）：{query_class}")

            prompt = f"<image>{prompt_template}\n\n" + "\n".join(prompt_parts)
            
            # 准备模型输入
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
            
            # 处理输入
            input_print = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenizer=False)
            print(f"模型输入准备完毕，{input_print}")
            inputs = self.prepare_inputs_for_vllm(messages, self.processor)
            print("Model input prepared, calling LLM...")

            # 调用模型
            outputs = self.llm.generate([inputs], sampling_params=self.sampling_params)
            generated_text = outputs[0].outputs[0].text
            print(f"Qwen-VL响应: {generated_text}")
            
            # 解析回答
            self._parse_qwen_response(generated_text, batch_detections)
        
        return detections
    
    def _parse_qwen_response(self, response: str, detections: List[Dict]):
        """
        解析Qwen-VL的响应
        
        Args:
            response: Qwen-VL的响应文本
            detections: 对应的检测框列表
        """
        # 清理响应文本
        response = response.strip()
        
        # 分割为行
        lines = response.split('\n')
        
        for i, det in enumerate(detections):
            region_key = f"区域{i+1}"
            region_found = False
            
            # 查找该区域的回答
            for line in lines:
                if line.strip().startswith(region_key):
                    region_found = True
                    self._parse_single_region_response(line, det)
                    break
            
            # 如果没找到该区域的回答
            if not region_found:
                det['qwen_verified'] = False
                det['qwen_final_class'] = det['class_name']
                det['qwen_response'] = "未找到对应区域的回答"
    
    def _parse_single_region_response(self, line: str, detection: Dict):
        """
        解析单个区域的回答
        
        Args:
            line: 回答行，如 "区域1：yes, 类别是person" 或 "区域1：no, 类别是car"
            detection: 检测框字典
        """
        detection['qwen_response'] = line.strip()
        
        # 提取yes/no和类别
        line_lower = line.lower()
        
        # 查找yes/no
        if 'yes' in line_lower:
            detection['qwen_verified'] = True
            detection['qwen_certainty'] = 'high'
            # 如果回答是yes，使用YOLO的类别
            detection['qwen_final_class'] = detection['class_name']
            
        elif 'no' in line_lower:
            detection['qwen_verified'] = False
            
            # 尝试提取Qwen-VL建议的类别
            # 查找"类别是"后面的内容
            pattern = r'类别是\s*([^,.\n]+)'
            matches = re.search(pattern, line)
            if matches:
                qwen_class = matches.group(1).strip()
                detection['qwen_final_class'] = qwen_class
                detection['qwen_certainty'] = 'medium'
            else:
                # 如果没有明确指定类别，尝试提取其他可能的类别描述
                words = line_lower.split()
                for word in words:
                    # 排除常见功能词
                    if word not in ['区域', ':', 'yes', 'no', '类别是', 'class', 'is', 'a', 'an', 'the']:
                        detection['qwen_final_class'] = word
                        detection['qwen_certainty'] = 'low'
                        break
                else:
                    detection['qwen_final_class'] = detection['class_name']
                    detection['qwen_certainty'] = 'unknown'
        else:
            # 无法确定是yes还是no
            detection['qwen_verified'] = False
            detection['qwen_final_class'] = detection['class_name']
            detection['qwen_certainty'] = 'unknown'
    
    def create_labelme_annotation(self, 
                                  image_path: str, 
                                  image: np.ndarray, 
                                  detections: List[Dict]) -> LabelmeAnnotation:
        """
        创建Labelme格式的标注
        
        Args:
            image_path: 图像路径
            image: 原始图像
            detections: 检测框列表
            
        Returns:
            LabelmeAnnotation对象
        """
        height, width = image.shape[:2]
        annotation = LabelmeAnnotation(image_path, (height, width, 3))
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # 创建属性字典
            attributes = {
                "yolo_class": det['class_name'],
                "yolo_confidence": f"{det['confidence']:.3f}",
                "yolo_class_id": str(det['class_id']),
                "qwen_verified": str(det['qwen_verified']).lower(),
                "qwen_final_class": det['qwen_final_class'],
                "qwen_certainty": det['qwen_certainty'] or "unknown",
            }
            
            if det['qwen_response']:
                attributes["qwen_response"] = det['qwen_response'][:100]  # 截断过长的响应
            
            # 添加形状标注
            # 使用Qwen-VL最终确定的类别作为标签
            label = det['qwen_final_class']
            
            annotation.add_shape(
                label=label,
                points=[[float(x1), float(y1)], [float(x2), float(y2)]],
                shape_type="rectangle",
                attributes=attributes,
                group_id=None
            )
        
        return annotation
    
    def visualize_results(self, 
                          image: np.ndarray, 
                          detections: List[Dict], 
                          output_path: str):
        """
        可视化结果并保存图像
        
        Args:
            image: 原始图像
            detections: 检测框列表
            output_path: 输出图像路径
        """
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # 根据验证结果选择颜色
            if det['qwen_verified']:
                color = (0, 255, 0)  # 绿色：验证通过
            else:
                color = (0, 0, 255)  # 红色：验证未通过
            
            # 绘制矩形框
            thickness = 2
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # 标签：YOLO类别 + Qwen-VL最终类别
            label = f"Y:{det['class_name']} Q:{det['qwen_final_class']}"
            if det['qwen_verified']:
                label += " ✓"
            else:
                label += " ✗"
            
            # 计算标签位置
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制标签背景
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_height - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # 保存可视化图像
        cv2.imwrite(output_path, vis_image)
        print(f"可视化结果已保存: {output_path}")
    
    def process_single_image(self, 
                             image_path: str, 
                             output_dir: str,
                             save_visualization: bool = True) -> bool:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            save_visualization: 是否保存可视化图像
            
        Returns:
            处理是否成功
        """
      
        print(f"\n处理图像: {image_path}")
        
        # 1. YOLOv8检测
        image, detections = self.detect_with_yolov8(image_path)
        print(f"  YOLOv8检测到 {len(detections)} 个目标")
        
        # 2. Qwen-VL二次研判
        if self.llm and detections:
            detections = self.batch_analyze_with_qwen_vl(image_path, detections)
            
            # 统计验证结果
            verified_count = sum(1 for d in detections if d['qwen_verified'])
            print(f"  Qwen-VL验证通过: {verified_count}/{len(detections)}")
        
        # 3. 创建Labelme标注
        annotation = self.create_labelme_annotation(image_path, image, detections)
        
        # 4. 保存Labelme JSON文件
        output_json_path = os.path.join(output_dir, f"{Path(image_path).stem}.json")
        annotation.save(output_json_path)
        print(f"  Labelme标注已保存: {output_json_path}")
        
        # 5. 保存可视化图像（可选）
        if save_visualization:
            vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_vis.jpg")
            self.visualize_results(image, detections, vis_path)
        
        return True
            

    
    def process_folder(self, 
                       input_folder: str, 
                       output_folder: str,
                       save_visualization: bool = True):
        """
        处理整个文件夹的图像
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            save_visualization: 是否保存可视化图像
        """
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG'}
        
        # 查找所有图像文件
        input_path = Path(input_folder)
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
        
        print(f"找到 {len(image_files)} 张图像")
        
        if not image_files:
            print("未找到图像文件!")
            return
        
        # 统计信息
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_detections': 0,
            'total_verified': 0
        }
        
        # 处理每张图像
        for image_file in tqdm(image_files, desc="处理图像"):
            stats['total'] += 1
            
            success = self.process_single_image(
                str(image_file),
                output_folder,
                save_visualization
            )
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        # 打印统计信息
        print("\n" + "="*50)
        print("处理完成!")
        print(f"总计图像: {stats['total']}")
        print(f"成功处理: {stats['success']}")
        print(f"处理失败: {stats['failed']}")
        
        # 保存处理统计信息
        stats_path = os.path.join(output_folder, "processing_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 将图像复制到输出文件夹
        for image_file in image_files:
            shutil.copy(image_file, output_folder)
        
        print(f"统计信息已保存: {stats_path}")

def main():
    """主函数"""
    # 配置参数
    config = {
        'input_folder': '/mnt/disk/lyf/images',  # 输入图像文件夹
        'output_folder': '/mnt/disk/lyf/labelme_annotations',  # 输出标注文件夹
        'yolo_model': '/mnt/disk/lyf/qwen/best.pt',  # YOLO模型路径
        'qwen_model': '/mnt/disk/lyf/ms-swift/output/rex_omni_labelme_full_data1_data2_support_neg_imgcap_improved/v9-20260126-204156/checkpoint-10000',  # Qwen-VL模型名称
        'confidence_threshold': 0.25,
        'iou_threshold': 0.45,
        'batch_size': 5,  # 批次大小
        'save_visualization': True  # 是否保存可视化图像
    }
    
    print("="*60)
    print("YOLOv8 + Qwen-VL 目标检测与二次研判系统")
    print("="*60)
    
    # 创建分析器
    analyzer = YOLOv8QwenVLAnalyzer(
        yolo_model_path=config['yolo_model'],
        qwen_model_name=config['qwen_model'],
        confidence_threshold=config['confidence_threshold'],
        iou_threshold=config['iou_threshold'],
        batch_size=config['batch_size']
    )
    
    # 处理文件夹
    analyzer.process_folder(
        input_folder=config['input_folder'],
        output_folder=config['output_folder'],
        save_visualization=config['save_visualization']
    )

def parse_single_image_example():
    """单张图像处理示例"""
    # 配置
    image_path = '/path/to/single/image.jpg'
    output_dir = './output'
    
    # 创建分析器
    analyzer = YOLOv8QwenVLAnalyzer(
        yolo_model_path='yolov8n.pt',
        qwen_model_name='Qwen/Qwen2-VL-2B-Instruct'
    )
    
    # 处理单张图像
    analyzer.process_single_image(image_path, output_dir, save_visualization=True)

if __name__ == "__main__":
    # 设置环境变量（如果需要使用镜像源）
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 运行主函数
    main()
    
    # 或者运行单张图像示例
    # parse_single_image_example()