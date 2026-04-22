# -*- coding: utf-8 -*-
import os
# 设置环境变量（根据您的环境调整）
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import json
import random
import base64
import glob
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import argparse
import torch
# Qwen3-VL 相关导入
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import ImageFont

class LabelmeVQAGenerator:
    """基于Qwen3-VL的Labelme VQA生成器"""
    
    def __init__(self, qwen_checkpoint_path="/mnt/data/lyf/Qwen3-VL-32B-Instruct"):
        """
        初始化Qwen3-VL模型
        
        Args:
            qwen_checkpoint_path: Qwen3-VL模型路径
        """
        print("正在加载Qwen3-VL模型...")
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
            temperature=0.7,
            max_tokens=9280,
            top_k=-1,
            stop_token_ids=[],
        )
        print("Qwen3-VL模型加载完成！")
    
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
    
    def process_single_labelme_file(self, json_file):
        """
        处理单个Labelme JSON文件
        
        Args:
            json_file: JSON文件路径
            
        Returns:
            tuple: (处理成功布尔值, 数据字典, 类别列表)
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            image_path = data['imagePath']
            base_dir = os.path.dirname(json_file)
            full_image_path = os.path.join(base_dir, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"警告: 图像文件不存在: {full_image_path}")
                return False, None, []
                
            # 加载图像
            try:
                image = Image.open(full_image_path)
            except Exception as e:
                print(f"无法加载图像 {full_image_path}: {e}")
                return False, None, []
                
            # 提取边界框和类别
            bboxes = []
            all_classes = []
            for shape in data['shapes']:
                if shape['shape_type'] == 'rectangle':
                    points = shape['points']
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    bbox = {
                        'label': shape['label'].split("/")[0],
                        'points': points,
                        'bbox': [min(x_coords), min(y_coords), 
                                max(x_coords), max(y_coords)],
                        'shape_type': 'rectangle',
                        'group_id': shape.get('group_id'),
                        'flags': shape.get('flags', {})
                    }
                    bboxes.append(bbox)
                    
                    # 收集类别
                    if shape['label'] not in all_classes:
                        all_classes.append(shape['label'])
            
            data_dict = {
                'json_file': json_file,
                'image_path': full_image_path,
                'image': image,
                'original_data': data,
                'bboxes': bboxes,
                'image_size': (data['imageWidth'], data['imageHeight'])
            }
            
            return True, data_dict, all_classes
            
        except Exception as e:
            print(f"处理JSON文件 {json_file} 时出错: {e}")
            return False, None, []
    
    def ask_qwen_about_region(self, image, bbox, true_label):
        """
        向Qwen3-VL询问关于特定区域的问题，生成负样本类别
        
        Args:
            image: PIL图像
            bbox: 边界框坐标 [x_min, y_min, x_max, y_max]
            true_label: 真实类别标签
            
        Returns:
            List: 负样本类别列表
        """
        try:
            width, height = image.size
            x_min, y_min, x_max, y_max = bbox
            
            # 转换为归一化坐标 [0, 1000]
            norm_x1 = int(x_min * 1000 / width)
            norm_y1 = int(y_min * 1000 / height)
            norm_x2 = int(x_max * 1000 / width)
            norm_y2 = int(y_max * 1000 / height)
            
            # 构建消息，让Qwen描述这个区域
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text", 
                            "text": f"我要做一个负样本问答对，给你一个矩形框，需要你首先告诉我矩形框包含的类别，然后需要你帮助我输出4个给定矩形框不包含的目标类别名称。仅输出伪类别即可。"
                                    f"输出格式为： 真实类别：xxx. \n 4个该矩形框不包含的目标类别名称（伪类别）：xxxx, xxxx, xxxx, xxxx \n"
                                    f"矩形框区域为[{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]的矩形框。"
                        }
                    ],
                }
            ]
            
            inputs = [self.prepare_qwen_inputs(messages)]
            outputs = self.qwen_llm.generate(inputs, sampling_params=self.qwen_sampling_params)
            # 从回复中提取伪类别，伪类别在"4个该矩形框不包含的目标类别名称（伪类别）："之后
            response = outputs[0].outputs[0].text.strip()
            pseudo_categories = []
            if "4个该矩形框不包含的目标类别名称（伪类别）：" in response:
                pseudo_categories = response.split("4个该矩形框不包含的目标类别名称（伪类别）：")[1].strip().split(", ")

            # 解析响应，提取类别
            categories = pseudo_categories
            
            print(f"区域 [{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}] - 真实标签: {true_label}")
            print(f"Qwen生成的负样本类别: {categories}")
            
            return categories[:4]  # 最多返回4个
            
        except Exception as e:
            print(f"Qwen区域询问失败: {e}")
            return []
    
    def generate_vqa_pairs_for_bbox(self, image, bbox_info, use_qwen=True):
        """
        为单个边界框生成VQA问答对
        
        Args:
            image: PIL图像
            bbox_info: 边界框信息
            use_qwen: 是否使用Qwen生成负样本
            
        Returns:
            List: VQA问答对列表
        """
        true_label = bbox_info['label']
        bbox = bbox_info['bbox']
        vqa_pairs = []
        
        # 1. 生成正样本
        positive_question = f"这个矩形框的类别是{true_label}吗？"
        positive_answer = f"yes, 类别是{true_label}"
        vqa_pairs.append({
            "question": positive_question,
            "answer": positive_answer
        })
        
        # 2. 生成负样本
        negative_categories = []
        
        if use_qwen:
            # 使用Qwen生成相关的负样本类别
            negative_categories = self.ask_qwen_about_region(image, bbox, true_label)

        
        # 确保负样本类别唯一
        negative_categories = list(dict.fromkeys(negative_categories))[:4]
        
        # 生成负样本问答对
        for neg_category in negative_categories:
            negative_question = f"这个矩形框的类别是{neg_category}吗？"
            negative_answer = f"no, 类别是{true_label}"
            vqa_pairs.append({
                "question": negative_question,
                "answer": negative_answer
            })
        return vqa_pairs
    
    def visualize_bbox_with_vqa(self, image, bboxes_with_vqa, output_path=None):
        """
        可视化带有VQA问答对的边界框
        
        Args:
            image: PIL图像
            bboxes_with_vqa: 带有VQA的边界框列表
            output_path: 输出路径（可选）
            
        Returns:
            PIL.Image: 可视化图像
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # 颜色列表
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
        ]
        
        for i, bbox_info in enumerate(bboxes_with_vqa):
            color = colors[i % len(colors)]
            
            points = bbox_info['points']
            label = bbox_info['label']
            
            # 绘制矩形框
            x_min, y_min = points[0]
            x_max, y_max = points[1]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
            
            # 添加标签
            label_text = f"{label}"
            try:
                font = ImageFont.truetype("arial.ttf", size=20)
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 标签背景
            draw.rectangle([x_min, y_min - text_height - 5, x_min + text_width + 10, y_min], 
                         fill=color)
            
            # 标签文本
            draw.text((x_min + 5, y_min - text_height - 2), label_text, 
                     fill=(255, 255, 255), font=font)
        
        if output_path:
            img.save(output_path)
            print(f"可视化结果已保存到: {output_path}")
        
        return img
    
    def generate_vqa_dataset_iterative(self, input_path, output_dir, use_qwen=True, 
                                     num_samples_per_bbox=5, visualize=True):
        """
        迭代生成VQA数据集（单张处理，节省内存）
        
        Args:
            input_path: 输入Labelme数据路径
            output_dir: 输出目录
            use_qwen: 是否使用Qwen生成负样本
            num_samples_per_bbox: 每个框的问答对数量
            visualize: 是否生成可视化图像
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # 获取所有JSON文件
        if os.path.isfile(input_path):
            json_files = [input_path]
        else:
            json_files = glob.glob(os.path.join(input_path, "*.json"))
        
        # 随机打乱文件顺序
        random.shuffle(json_files)
        
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 统计信息
        total_images = 0
        total_bboxes = 0
        total_vqa_pairs = 0
        all_classes_set = set()
        
        # 处理每个JSON文件
        for json_file in json_files:
            try:
                # 处理单个文件
                success, item, file_classes = self.process_single_labelme_file(json_file)
                
                if not success or not item:
                    continue
                    
                image = item['image']
                original_data = item['original_data']
                bboxes = item['bboxes']
                image_filename = os.path.basename(item['image_path'])
                output_image_path = os.path.join(output_dir, "images", image_filename)
                
                # 检查是否已处理过
                if os.path.exists(output_image_path):
                    print(f"已存在输出图像: {output_image_path}, skip")
                    continue
                
                # 更新类别集合
                for cls in file_classes:
                    all_classes_set.add(cls)
                
                # 更新形状列表，添加VQA
                updated_shapes = []
                bboxes_with_vqa = []
                
                for bbox_info in bboxes:
                    # 生成VQA问答对
                    vqa_pairs = self.generate_vqa_pairs_for_bbox(
                        image, bbox_info, use_qwen
                    )
                    
                    # 更新形状信息，添加bboxvqa字段
                    updated_shape = {
                        "label": bbox_info['label'],
                        "points": bbox_info['points'],
                        "shape_type": bbox_info['shape_type'],
                        "group_id": bbox_info.get('group_id'),
                        "flags": bbox_info.get('flags', {}),
                        "bboxvqa": vqa_pairs
                    }
                    updated_shapes.append(updated_shape)
                    bboxes_with_vqa.append(bbox_info)
                    
                    # 更新统计
                    total_bboxes += 1
                    total_vqa_pairs += len(vqa_pairs)
                
                # 更新原始数据
                original_data['shapes'] = updated_shapes
                
                # 保存图像
                image.save(output_image_path)
                original_data['imagePath'] = os.path.join("images", image_filename)
                
                # 保存JSON
                json_filename = os.path.splitext(image_filename)[0] + '.json'
                json_dir = os.path.join(output_dir, 'jsons')
                os.makedirs(json_dir, exist_ok=True)
                json_path = os.path.join(json_dir, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, ensure_ascii=False, indent=2)
                
                # 生成可视化
                if visualize:
                    vis_filename = os.path.splitext(image_filename)[0] + '_vqa.jpg'
                    vis_path = os.path.join(output_dir, vis_filename)
                    self.visualize_bbox_with_vqa(image, bboxes_with_vqa, vis_path)
                
                total_images += 1
                print(f"已处理: {image_filename} - {len(bboxes)}个边界框")
                
                # 每处理10张图像输出一次进度
                if total_images % 10 == 0:
                    print(f"进度: 已处理 {total_images}/{len(json_files)} 张图像")
                
                # 释放内存
                del image
                
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
                continue
        
        # 保存数据集信息
        dataset_info = {
            "total_images": total_images,
            "total_bboxes": total_bboxes,
            "total_vqa_pairs": total_vqa_pairs,
            "classes": list(all_classes_set),
            "use_qwen": use_qwen,
            "num_samples_per_bbox": num_samples_per_bbox
        }
        
        info_path = os.path.join(output_dir, "dataset_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 数据集生成完成 ===")
        print(f"输出目录: {output_dir}")
        print(f"图像数量: {total_images}")
        print(f"边界框数量: {total_bboxes}")
        print(f"VQA问答对总数: {total_vqa_pairs}")
        print(f"是否使用Qwen: {use_qwen}")
        print(f"数据集信息已保存到: {info_path}")
    
    def generate_vqa_dataset(self, input_path, output_dir, use_qwen=True, 
                           num_samples_per_bbox=5, visualize=True):
        """
        保持向后兼容的生成方法（调用迭代版本）
        """
        return self.generate_vqa_dataset_iterative(
            input_path, output_dir, use_qwen, num_samples_per_bbox, visualize
        )


def main():
    parser = argparse.ArgumentParser(description='基于Qwen3-VL的Labelme VQA数据集生成器')
    parser.add_argument('--input', type=str, default='/mnt/disk/lyf/datasets/needed/labelme_train', help='Labelme输入路径（文件或目录）')
    parser.add_argument('--output', type=str, default='/mnt/disk/lyf/datasets/needed/qwen_objvqa', help='输出目录')
    parser.add_argument('--qwen-path', type=str, default='/mnt/disk/lyf/Qwen3-VL-32B-Instruct', 
                       help='Qwen3-VL模型路径')
    parser.add_argument('--use-qwen', action='store_true', default=True, 
                       help='使用Qwen3-VL生成负样本（默认启用）')
    parser.add_argument('--no-qwen', dest='use_qwen', action='store_false',
                       help='不使用Qwen3-VL，仅使用随机负样本')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='每个边界框的问答对数量（默认5）')
    parser.add_argument('--visualize', action='store_true', default=False,
                       help='生成可视化图像（默认启用）')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                       help='不生成可视化图像')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认42）')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("初始化VQA生成器...")
    generator = LabelmeVQAGenerator(qwen_checkpoint_path=args.qwen_path)
    
    print(f"开始生成VQA数据集:")
    print(f"  输入路径: {args.input}")
    print(f"  输出目录: {args.output}")
    print(f"  使用Qwen: {args.use_qwen}")
    print(f"  每个框样本数: {args.num_samples}")
    print(f"  迭代处理模式: 是（节省内存）")
    
    generator.generate_vqa_dataset_iterative(
        input_path=args.input,
        output_dir=args.output,
        use_qwen=args.use_qwen,
        num_samples_per_bbox=args.num_samples,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()