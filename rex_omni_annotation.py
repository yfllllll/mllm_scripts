#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
  
import os  
import json  
import argparse  
from pathlib import Path  
from typing import List, Dict, Any, Tuple  
from PIL import Image  
from rex_omni import RexOmniWrapper  
  
def resize_image_with_aspect_ratio(
    image: Image.Image, 
    max_size: int = 1260
) -> Tuple[Image.Image, float]:
    """
    将图像等比缩放到指定最大边长
    
    Args:
        image: 输入图像
        max_size: 最大边长
        
    Returns:
        (缩放后的图像, 缩放比例)
    """
    width, height = image.size
    
    # 计算缩放比例
    if width > height:
        scale = max_size / width
        new_width = max_size
        new_height = int(height * scale)
    else:
        scale = max_size / height
        new_height = max_size
        new_width = int(width * scale)
    
    # 使用LANCZOS重采样进行高质量缩放
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    print(f"图像缩放: {width}x{height} -> {new_width}x{new_height} (缩放比例: {scale:.4f})")
    return resized_image, scale

def scale_coordinates_back(
    coords: List[float], 
    scale: float, 
    coord_type: str
) -> List[float]:
    """
    将坐标从缩放后的图像尺度还原到原始尺度
    
    Args:
        coords: 坐标列表
        scale: 缩放比例
        coord_type: 坐标类型 ('box', 'point', 'polygon')
        
    Returns:
        还原后的坐标列表
    """
    if coord_type == "box":
        # 边界框: [x1, y1, x2, y2]
        return [coord / scale for coord in coords]
    elif coord_type == "point":
        # 点: [x, y]
        return [coord / scale for coord in coords]
    elif coord_type == "polygon":
        # 多边形: [x1, y1, x2, y2, ...]
        return [coord / scale for coord in coords]
    else:
        raise ValueError(f"未知的坐标类型: {coord_type}")
  
def create_labelme_annotation(  
    image_path: str,  
    predictions: Dict[str, List[Dict]],  
    image_width: int,  
    image_height: int,  
    existing_data: Dict[str, Any] = None 
) -> Dict[str, Any]:  
    """  
    将Rex-Omni的预测结果转换为labelme格式的JSON  
      
    Args:  
        image_path: 图像文件路径  
        predictions: Rex-Omni的预测结果  
        image_width: 图像宽度  
        image_height: 图像高度  
        existing_data: 已存在的标注数据（如果存在）
      
    Returns:  
        labelme格式的标注数据  
    """  
    if existing_data:
        # 使用已存在的数据作为基础
        labelme_data = existing_data
        # 更新图像信息（以防图像有变化）
        labelme_data["imagePath"] = os.path.basename(image_path)
        labelme_data["imageHeight"] = image_height
        labelme_data["imageWidth"] = image_width
    else:
        # 创建新的标注数据结构
        labelme_data = {  
            "version": "5.0.1",  
            "flags": {},  
            "shapes": [],  
            "imagePath": os.path.basename(image_path),  
            "imageData": None,  # labelme通常不包含图像数据  
            "imageHeight": image_height,  
            "imageWidth": image_width  
        }  
      
    # 遍历所有类别的预测结果  
    for category, detections in predictions.items():  
        for detection in detections:  
            if detection["type"] == "box":  
                # 边界框格式转换  
                coords = detection["coords"]  
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]  
                  
                shape = {  
                    "label": category,  
                    "points": [[x1, y1], [x2, y2]],  
                    "group_id": None,  
                    "shape_type": "rectangle",  
                    "flags": {}  
                }  
                labelme_data["shapes"].append(shape)  
                  
            elif detection["type"] == "point":  
                # 点标注格式  
                coords = detection["coords"]  
                x, y = coords[0], coords[1]  
                  
                shape = {  
                    "label": category,  
                    "points": [[x, y]],  
                    "group_id": None,  
                    "shape_type": "point",  
                    "flags": {}  
                }  
                labelme_data["shapes"].append(shape)  
                  
            elif detection["type"] == "polygon":  
                # 多边形格式  
                coords = detection["coords"]  
                # 将坐标列表转换为点对  
                points = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]  
                  
                shape = {  
                    "label": category,  
                    "points": points,  
                    "group_id": None,  
                    "shape_type": "polygon",  
                    "flags": {}  
                }  
                labelme_data["shapes"].append(shape)  
      
    return labelme_data  

def load_existing_annotation(json_path: str) -> Dict[str, Any]:
    """
    加载已存在的标注文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        已存在的标注数据，如果文件不存在或格式错误则返回None
    """
    if not os.path.exists(json_path):
        return None
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # 验证基本结构
        if isinstance(existing_data, dict) and "shapes" in existing_data:
            print(f"加载已存在的标注文件: {json_path}")
            return existing_data
        else:
            print(f"警告: {json_path} 格式不正确，将创建新的标注文件")
            return None
            
    except Exception as e:
        print(f"加载 {json_path} 时出错: {e}，将创建新的标注文件")
        return None
  
def process_single_image(  
    rex_model: RexOmniWrapper,  
    image_path: str,  
    categories: List[str],  
    task: str = "detection",  
    output_dir: str = None,
    overwrite: bool = False,
    relative_path: str = None,
    resize_max_size: int = 1260
) -> str:  
    """  
    处理单张图像并生成labelme标注文件  
      
    Args:  
        rex_model: Rex-Omni模型实例  
        image_path: 图像文件路径  
        categories: 检测类别列表  
        task: 任务类型  
        output_dir: 输出目录，如果为None则使用图像所在目录
        overwrite: 是否覆盖已存在的标注（False时追加）
        relative_path: 相对于输入目录的相对路径，用于保持目录结构
        resize_max_size: 图像resize的最大边长
        
    Returns:  
        生成的JSON文件路径  
    """  
    # 加载图像  
    original_image = Image.open(image_path).convert("RGB")  
    original_width, original_height = original_image.size  
      
    print(f"处理图像: {image_path} (原始尺寸: {original_width}x{original_height})")  
    
    # 等比resize图像
    resized_image, scale = resize_image_with_aspect_ratio(original_image, resize_max_size)
    resized_width, resized_height = resized_image.size
      
    # 执行推理（使用resize后的图像）
    results = rex_model.inference(  
        images=resized_image,  
        task=task,  
        categories=categories  
    )  
      
    result = results[0]  
    if not result.get("success", False):  
        raise RuntimeError(f"推理失败: {result.get('error', '未知错误')}")  
      
    predictions = result["extracted_predictions"]  
    
    # 将坐标还原到原始尺度
    scaled_predictions = {}
    for category, detections in predictions.items():
        scaled_detections = []
        for detection in detections:
            scaled_detection = detection.copy()
            # 还原坐标
            scaled_detection["coords"] = scale_coordinates_back(
                detection["coords"], 
                scale, 
                detection["type"]
            )
            scaled_detections.append(scaled_detection)
        scaled_predictions[category] = scaled_detections
    
    # 统计检测结果  
    total_detections = sum(len(dets) for dets in scaled_predictions.values())  
    print(f"检测到 {total_detections} 个对象 (已还原到原始尺度):")  
    for cat, dets in scaled_predictions.items():  
        print(f"  - {cat}: {len(dets)} 个")  
      
    # 确定输出路径
    if output_dir is None:  
        output_dir = os.path.dirname(image_path)
        json_output_dir = output_dir
    else:
        if relative_path:
            # 保持目录结构
            json_output_dir = os.path.join(output_dir, relative_path)
        else:
            json_output_dir = output_dir
      
    os.makedirs(json_output_dir, exist_ok=True)  
      
    # 生成JSON文件名  
    image_name = Path(image_path).stem  
    json_path = os.path.join(json_output_dir, f"{image_name}.json")  
    
    # 检查是否已存在标注文件
    existing_data = None
    if not overwrite and os.path.exists(json_path):
        existing_data = load_existing_annotation(json_path)
        if existing_data:
            existing_shapes_count = len(existing_data.get("shapes", []))
            print(f"已存在 {existing_shapes_count} 个标注，将追加新的检测结果")
      
    # 转换为labelme格式（使用原始图像尺寸和还原后的坐标）
    labelme_data = create_labelme_annotation(  
        image_path, scaled_predictions, original_width, original_height, existing_data
    )  
      
    # 保存JSON文件  
    with open(json_path, 'w', encoding='utf-8') as f:  
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)  
      
    final_shapes_count = len(labelme_data.get("shapes", []))
    print(f"标注文件已保存: {json_path} (共 {final_shapes_count} 个标注)")  
    return json_path  

def find_image_files_recursive(
    image_dir: str,
    image_extensions: List[str] = None
) -> List[tuple]:
    """
    递归查找所有图像文件，返回文件路径和相对于image_dir的相对路径
    
    Args:
        image_dir: 图像目录路径
        image_extensions: 支持的图像扩展名
        
    Returns:
        列表，每个元素为(图像绝对路径, 相对于image_dir的相对路径)
    """
    if image_extensions is None:  
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_dir_path = Path(image_dir).resolve()
    image_files = []
    
    # 递归查找所有图像文件
    for ext in image_extensions:
        pattern = f"**/*{ext}"
        for file_path in image_dir_path.glob(pattern):
            if file_path.is_file():
                # 计算相对于根目录的相对路径
                relative_path = file_path.relative_to(image_dir_path).parent
                image_files.append((str(file_path), str(relative_path)))
        
        # 同时查找大写扩展名
        pattern_upper = f"**/*{ext.upper()}"
        for file_path in image_dir_path.glob(pattern_upper):
            if file_path.is_file():
                relative_path = file_path.relative_to(image_dir_path).parent
                image_files.append((str(file_path), str(relative_path)))
    
    return image_files
  
def batch_process_images(  
    image_dir: str,  
    categories: List[str],  
    model_path: str = "IDEA-Research/Rex-Omni",  
    backend: str = "transformers",  
    task: str = "detection",  
    output_dir: str = None,  
    image_extensions: List[str] = None,
    overwrite: bool = False,
    resize_max_size: int = 1260
):  
    """  
    批量处理图像目录中的所有图像（包括子文件夹）
      
    Args:  
        image_dir: 图像目录路径  
        categories: 检测类别列表  
        model_path: Rex-Omni模型路径  
        backend: 推理后端  
        task: 任务类型  
        output_dir: 输出目录，如果为None则使用图像目录  
        image_extensions: 支持的图像扩展名
        overwrite: 是否覆盖已存在的标注
        resize_max_size: 图像resize的最大边长
    """  
    # 初始化Rex-Omni模型  
    print(f"初始化Rex-Omni模型: {model_path}")  
    rex_model = RexOmniWrapper(  
        model_path=model_path,  
        backend=backend,  
        max_tokens=2048,  
        temperature=0.0  
    )  
      
    # 递归查找所有图像文件
    print("正在递归查找图像文件...")
    image_files = find_image_files_recursive(image_dir, image_extensions)
      
    print(f"找到 {len(image_files)} 个图像文件")  
    print(f"图像将等比resize到最大边长: {resize_max_size}")
      
    if output_dir is None:  
        output_dir = image_dir  
    
    # 创建输出根目录
    os.makedirs(output_dir, exist_ok=True)
      
    # 批量处理  
    success_count = 0
    skipped_count = 0
    for image_path, relative_path in image_files:  
        try:  
            # 检查是否已存在标注文件（如果设置了不覆盖且文件存在）
            image_name = Path(image_path).stem
            json_output_dir = os.path.join(output_dir, relative_path) if relative_path != "." else output_dir
            json_path = os.path.join(json_output_dir, f"{image_name}.json")
            
            if not overwrite and os.path.exists(json_path):
                print(f"跳过已存在的标注文件: {json_path}")
                skipped_count += 1
                continue
                
            process_single_image(  
                rex_model=rex_model,  
                image_path=image_path,  
                categories=categories,  
                task=task,  
                output_dir=output_dir,
                overwrite=overwrite,
                relative_path=relative_path,
                resize_max_size=resize_max_size
            )  
            success_count += 1  
        except Exception as e:  
            print(f"处理 {image_path} 时出错: {e}")  
      
    print(f"\n批量处理完成!")  
    print(f"成功处理: {success_count} 个图像")  
    print(f"跳过已存在: {skipped_count} 个图像")  
    print(f"总计找到: {len(image_files)} 个图像文件")  
  
def main():  
    parser = argparse.ArgumentParser(description="Rex-Omni数据标注工具 - 生成labelme格式的JSON标注文件")  
      
    parser.add_argument("--image_path", type=str, help="单个图像文件路径")  
    parser.add_argument("--image_dir", type=str, help="图像目录路径（批量处理)", default='/mnt/data/lyf/datasets/animals/animal460') 
    parser.add_argument("--categories", type=str, default="person,vehicles",   
                       help="检测类别，用逗号分隔，例如: person,car,dog")  
    parser.add_argument("--model_path", type=str, default="/mnt/data/lyf/IDEA-Research/Rex-Omni",  
                       help="Rex-Omni模型路径")  
    parser.add_argument("--backend", type=str, default="vllm",   
                       choices=["transformers", "vllm"],  
                       help="推理后端")  
    parser.add_argument("--task", type=str, default="detection",  
                       choices=["detection", "pointing", "ocr_box", "ocr_polygon"],  
                       help="任务类型")  
    parser.add_argument("--output_dir", type=str, default='/mnt/data/lyf/datasets/detect_base/tmp',  
                       help="输出目录（默认为图像所在目录）")
    parser.add_argument("--overwrite", action="store_false",
                       help="覆盖已存在的标注文件（默认是追加模式）")  
    parser.add_argument("--recursive", action="store_true", default=True,
                       help="递归处理子文件夹（默认启用）")  
    parser.add_argument("--resize_max_size", type=int, default=1260,
                       help="图像等比resize的最大边长（默认1260）")
      
    args = parser.parse_args()  
      
    # 解析类别列表  
    categories = [cat.strip() for cat in args.categories.split(",")]  
    print(f"检测类别: {categories}")  
    
    if args.overwrite:
        print("模式: 覆盖已存在的标注文件")
    else:
        print("模式: 追加到已存在的标注文件")
    
    print(f"递归处理: {'启用' if args.recursive else '禁用'}")
    print(f"图像resize最大边长: {args.resize_max_size}")
      
    if args.image_path:  
        # 单个图像处理  
        if not os.path.exists(args.image_path):  
            print(f"错误: 图像文件不存在: {args.image_path}")  
            return  
          
        # 初始化模型  
        rex_model = RexOmniWrapper(  
            model_path=args.model_path,  
            backend=args.backend,  
            max_tokens=4096,  
            temperature=0.0  
        )  
          
        try:  
            process_single_image(  
                rex_model=rex_model,  
                image_path=args.image_path,  
                categories=categories,  
                task=args.task,  
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                resize_max_size=args.resize_max_size
            )  
        except Exception as e:  
            print(f"处理失败: {e}")  
      
    elif args.image_dir:  
        # 批量处理  
        if not os.path.exists(args.image_dir):  
            print(f"错误: 图像目录不存在: {args.image_dir}")  
            return  
          
        batch_process_images(  
            image_dir=args.image_dir,  
            categories=categories,  
            model_path=args.model_path,  
            backend=args.backend,  
            task=args.task,  
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            resize_max_size=args.resize_max_size
        )  
      
    else:  
        print("错误: 请指定 --image_path 或 --image_dir")  
        parser.print_help()  
  
if __name__ == "__main__":  
    main()