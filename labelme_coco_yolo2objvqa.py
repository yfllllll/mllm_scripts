import json
import os
import random
import glob
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any
import cv2
import numpy as np

class VQADatasetConverter:
    def __init__(self, num_negative_samples: int = 4):
        """
        初始化VQA数据集转换器
        
        Args:
            num_negative_samples: 每个框的负样本数量
        """
        self.num_negative_samples = num_negative_samples
        self.classes = []  # 存储数据集的所有类别
        
    def load_labelme_data(self, input_path: str) -> Tuple[Dict[str, List[Dict]], List[str]]:
        """
        加载Labelme格式数据
        
        Args:
            input_path: 输入路径（可以是单个json文件或包含json的文件夹）
            
        Returns:
            Tuple[Dict, List]: (图像数据字典, 类别列表)
        """
        data_dict = {}
        
        if os.path.isfile(input_path):
            json_files = [input_path]
            base_dir = os.path.dirname(input_path)
        else:
            json_files = glob.glob(os.path.join(input_path, "*.json"))
            base_dir = input_path
            
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            image_path = data['imagePath']
            full_image_path = os.path.join(base_dir, image_path)
            
            # 收集所有类别
            for shape in data['shapes']:
                label = shape['label']
                if label not in self.classes:
                    self.classes.append(label)
                    
            # 提取边界框
            bboxes = []
            for shape in data['shapes']:
                if shape['shape_type'] == 'rectangle':
                    points = shape['points']
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    bbox = {
                        'label': shape['label'],
                        'points': points,
                        'bbox': [min(x_coords), min(y_coords), 
                                max(x_coords), max(y_coords)]
                    }
                    bboxes.append(bbox)
                    
            data_dict[json_file] = {
                'image_path': full_image_path,
                'data': data,
                'bboxes': bboxes
            }
            
        return data_dict, self.classes
    
    def load_coco_data(self, annotation_file: str, image_dir: str) -> Tuple[Dict[str, List[Dict]], List[str]]:
        """
        加载COCO格式数据
        
        Args:
            annotation_file: COCO标注文件路径
            image_dir: 图像目录
            
        Returns:
            Tuple[Dict, List]: (图像数据字典, 类别列表)
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            
        data_dict = {}
        
        # 构建ID到名称的映射
        category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        image_id_to_info = {img['id']: img for img in coco_data['images']}
        
        # 收集所有类别
        for cat in coco_data['categories']:
            if cat['name'] not in self.classes:
                self.classes.append(cat['name'])
                
        # 按图像分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
            
        # 处理每张图像
        for image_id, anns in annotations_by_image.items():
            img_info = image_id_to_info[image_id]
            image_path = os.path.join(image_dir, img_info['file_name'])
            
            bboxes = []
            for ann in anns:
                # COCO格式: [x, y, width, height]
                x, y, w, h = ann['bbox']
                points = [[x, y], [x + w, y + h]]
                
                bbox = {
                    'label': category_id_to_name[ann['category_id']],
                    'points': points,
                    'bbox': [x, y, x + w, y + h]
                }
                bboxes.append(bbox)
                
            # 构建类似Labelme的数据结构
            labelme_data = {
                "version": "5.1.1",
                "flags": {},
                "shapes": [],
                "imagePath": img_info['file_name'],
                "imageData": None,
                "imageHeight": img_info['height'],
                "imageWidth": img_info['width']
            }
            
            data_dict[image_path] = {
                'image_path': image_path,
                'data': labelme_data,
                'bboxes': bboxes
            }
            
        return data_dict, self.classes
    
    def load_yolo_data(self, data_dir: str, classes_file: str = None) -> Tuple[Dict[str, List[Dict]], List[str]]:
        """
        加载YOLO格式数据
        
        Args:
            data_dir: 数据目录（包含images和labels子目录）
            classes_file: 类别文件路径（可选）
            
        Returns:
            Tuple[Dict, List]: (图像数据字典, 类别列表)
        """
        data_dict = {}
        
        # 确定目录结构
        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')
        
        if not os.path.exists(images_dir):
            # 尝试直接在当前目录查找图像
            images_dir = data_dir
            
        if not os.path.exists(labels_dir):
            labels_dir = data_dir
            
        # 加载类别
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            # 尝试从目录中自动发现类别
            self.classes = []
            
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
            
        for image_file in image_files:
            # 对应的标签文件
            label_file = os.path.join(
                labels_dir, 
                os.path.splitext(os.path.basename(image_file))[0] + '.txt'
            )
            
            if not os.path.exists(label_file):
                continue
                
            # 读取图像尺寸
            img = cv2.imread(image_file)
            if img is None:
                continue
            height, width = img.shape[:2]
            
            bboxes = []
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    class_id = int(parts[0])
                    # YOLO格式: [center_x, center_y, width, height] 归一化坐标
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # 转换为绝对坐标
                    x_center = cx * width
                    y_center = cy * height
                    bbox_width = w * width
                    bbox_height = h * height
                    
                    # 转换为矩形坐标 [x_min, y_min, x_max, y_max]
                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2
                    x_max = x_center + bbox_width / 2
                    y_max = y_center + bbox_height / 2
                    
                    # 确保坐标在图像范围内
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width, x_max)
                    y_max = min(height, y_max)
                    
                    # 使用类别ID或尝试获取类别名称
                    if class_id < len(self.classes):
                        label = self.classes[class_id]
                    else:
                        label = f"class_{class_id}"
                        
                    points = [[x_min, y_min], [x_max, y_max]]
                    
                    bbox = {
                        'label': label,
                        'points': points,
                        'bbox': [x_min, y_min, x_max, y_max]
                    }
                    bboxes.append(bbox)
                    
            # 构建类似Labelme的数据结构
            labelme_data = {
                "version": "5.1.1",
                "flags": {},
                "shapes": [],
                "imagePath": os.path.basename(image_file),
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width
            }
            
            data_dict[image_file] = {
                'image_path': image_file,
                'data': labelme_data,
                'bboxes': bboxes
            }
            
        return data_dict, self.classes
    
    def generate_vqa_pairs(self, true_label: str) -> List[Dict[str, str]]:
        """
        为给定的真实标签生成VQA问答对
        
        Args:
            true_label: 真实类别标签
            
        Returns:
            List[Dict]: 包含问答对的列表
        """
        vqa_pairs = []
        
        # 生成正样本
        positive_question = f"这个矩形框的类别是{true_label}吗？"
        positive_answer = f"yes, 类别是{true_label}"
        vqa_pairs.append({
            "question": positive_question,
            "answer": positive_answer
        })
        
        # 生成负样本
        other_classes = [c for c in self.classes if c != true_label]
        
        # 如果类别不足，使用占位符
        if len(other_classes) < self.num_negative_samples:
            # 补充一些通用类别
            generic_classes = ["动物", "植物", "车辆", "建筑", "工具", "食物", "家具", "电器"]
            for gc in generic_classes:
                if gc not in self.classes:
                    other_classes.append(gc)
                    
        # 随机选择负样本类别
        if len(other_classes) > 0:
            negative_classes = random.sample(
                other_classes, 
                min(self.num_negative_samples, len(other_classes))
            )
            
            for neg_class in negative_classes:
                negative_question = f"这个矩形框的类别是{neg_class}吗？"
                negative_answer = f"no,类别是{true_label}"
                vqa_pairs.append({
                    "question": negative_question,
                    "answer": negative_answer
                })
                
        return vqa_pairs
    
    def convert_to_labelme_with_vqa(self, input_data: Dict, output_dir: str):
        """
        将输入数据转换为带VQA的Labelme格式
        
        Args:
            input_data: 输入数据字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for file_key, item in input_data.items():
            labelme_data = item['data']
            bboxes = item['bboxes']
            image_path = item['image_path']
            
            # 复制图像到输出目录
            image_filename = os.path.basename(image_path)
            output_image_path = os.path.join(output_dir, image_filename)
            
            if os.path.exists(image_path):
                shutil.copy2(image_path, output_image_path)
            
            # 为每个边界框生成VQA问答对并添加到Labelme格式
            shapes = []
            for bbox in bboxes:
                vqa_pairs = self.generate_vqa_pairs(bbox['label'])
                
                shape = {
                    "label": bbox['label'],
                    "points": bbox['points'],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                    "bboxvqa": vqa_pairs
                }
                shapes.append(shape)
            
            # 更新数据
            labelme_data['shapes'] = shapes
            labelme_data['imagePath'] = image_filename
            
            # 保存JSON文件
            json_filename = os.path.splitext(image_filename)[0] + '.json'
            json_path = os.path.join(output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)
                
            print(f"已处理: {image_filename}")
    
    def convert(self, input_path: str, output_dir: str, 
                input_format: str = 'auto', 
                coco_image_dir: str = None,
                yolo_classes_file: str = None):
        """
        主转换函数
        
        Args:
            input_path: 输入路径
            output_dir: 输出目录
            input_format: 输入格式 ('labelme', 'coco', 'yolo', 或 'auto')
            coco_image_dir: COCO格式的图像目录（仅当format='coco'时需要）
            yolo_classes_file: YOLO格式的类别文件（可选）
        """
        print(f"开始转换: {input_path}")
        
        # 自动检测格式
        if input_format == 'auto':
            if input_path.endswith('.json'):
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'images' in data and 'annotations' in data and 'categories' in data:
                        input_format = 'coco'
                    else:
                        input_format = 'labelme'
                except:
                    input_format = 'labelme'
            elif os.path.isdir(input_path):
                # 检查目录结构
                if os.path.exists(os.path.join(input_path, 'images')) and \
                   os.path.exists(os.path.join(input_path, 'labels')):
                    input_format = 'yolo'
                else:
                    input_format = 'labelme'
        
        # 根据格式加载数据
        if input_format == 'labelme':
            data_dict, classes = self.load_labelme_data(input_path)
        elif input_format == 'coco':
            if not coco_image_dir:
                coco_image_dir = os.path.dirname(input_path)
            data_dict, classes = self.load_coco_data(input_path, coco_image_dir)
        elif input_format == 'yolo':
            data_dict, classes = self.load_yolo_data(input_path, yolo_classes_file)
        else:
            raise ValueError(f"不支持的格式: {input_format}")
        
        print(f"检测到 {len(classes)} 个类别: {classes}")
        print(f"处理 {len(data_dict)} 个图像文件")
        
        # 转换数据
        self.convert_to_labelme_with_vqa(data_dict, output_dir)
        
        print(f"转换完成！输出目录: {output_dir}")
        
        # 生成数据集统计信息
        total_vqa_pairs = 0
        total_bboxes = 0
        for file_key, item in data_dict.items():
            total_bboxes += len(item['bboxes'])
            for bbox in item['bboxes']:
                total_vqa_pairs += 1 + self.num_negative_samples  # 1正 + 4负
        
        print(f"总计: {total_bboxes} 个边界框, {total_vqa_pairs} 个问答对")


def main():
    parser = argparse.ArgumentParser(description='将标注数据转换为带VQA问答对的Labelme格式')
    parser.add_argument('--input', default='/mnt/disk/lyf/datasets/coco/annotations/instances_train2017.json', type=str, help='输入路径（文件或目录）')
    parser.add_argument('--output', default='/mnt/disk/lyf/datasets/coco/obj_vqa', type=str, help='输出目录')
    parser.add_argument('--format', type=str, default='coco',
                       choices=['auto', 'labelme', 'coco', 'yolo'],
                       help='输入数据格式（默认：自动检测）')
    parser.add_argument('--coco_image_dir', default='/mnt/disk/lyf/datasets/coco/train2017', type=str,
                       help='COCO格式的图像目录（仅当format=coco时需要）')
    parser.add_argument('--yolo-classes', type=str,
                       help='YOLO格式的类别文件路径')
    parser.add_argument('--negative-samples', type=int, default=4,
                       help='每个框的负样本数量（默认：4）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认：42）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建转换器
    converter = VQADatasetConverter(num_negative_samples=args.negative_samples)
    
    # 执行转换
    converter.convert(
        input_path=args.input,
        output_dir=args.output,
        input_format=args.format,
        coco_image_dir=args.coco_image_dir,
        yolo_classes_file=args.yolo_classes
    )


if __name__ == "__main__":
    main()