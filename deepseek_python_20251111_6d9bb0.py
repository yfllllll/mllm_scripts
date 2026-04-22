import os
import json
import shutil
from tqdm import tqdm

def convert_labelme_files(source_dir, target_dir):
    """
    转换同一文件夹内的LabelMe标注文件和图像文件
    - source_dir: 源目录，包含图像文件和LabelMe标注文件
    - target_dir: 目标目录，将包含转换后的文件
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 收集所有JSON标注文件
    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON annotation files")
    
    # 处理每个标注文件
    valid_count = 0
    skipped_count = 0
    
    for json_file in tqdm(json_files, desc="Processing files"):
        json_path = os.path.join(source_dir, json_file)
        
        # 检查标注文件是否有效（非空且有标注）
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有有效的标注形状
            if not data.get('shapes') or len(data['shapes']) == 0:
                skipped_count += 1
                continue
                
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            skipped_count += 1
            continue
        
        # 获取对应的图像文件名
        image_filename = data.get('imagePath')
        if not image_filename:
            # 如果JSON中没有imagePath，尝试使用同名的图像文件
            base_name = os.path.splitext(json_file)[0]
            # 尝试常见图像扩展名
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                possible_image = base_name + ext
                if os.path.exists(os.path.join(source_dir, possible_image)):
                    image_filename = possible_image
                    break
        
        if not image_filename:
            print(f"No corresponding image found for {json_file}")
            skipped_count += 1
            continue
        
        image_path = os.path.join(source_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            skipped_count += 1
            continue
        
        # 转换标注文件编码
        target_json_path = os.path.join(target_dir, json_file)
        try:
            with open(target_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error converting {json_file}: {e}")
            skipped_count += 1
            continue
        
        # 复制图像文件
        target_image_path = os.path.join(target_dir, image_filename)
        shutil.copy2(image_path, target_image_path)
        
        valid_count += 1
    
    print(f"Processing completed:")
    print(f"  Valid files with annotations: {valid_count}")
    print(f"  Skipped files (no valid annotations): {skipped_count}")
    print(f"  Output directory: {target_dir}")

def batch_convert_folders(source_dirs, target_base_dir):
    """批量处理多个文件夹"""
    for source_dir in source_dirs:
        print(f"\nProcessing folder: {source_dir}")
        folder_name = os.path.basename(source_dir.rstrip('/'))
        target_dir = os.path.join(target_base_dir, f"{folder_name}_converted")
        
        if not os.path.exists(source_dir):
            print(f"Source directory does not exist: {source_dir}")
            continue
            
        convert_labelme_files(source_dir, target_dir)

if __name__ == "__main__":
    # 使用方法示例：
    
    # 方法1: 处理单个文件夹
    source_dir = "/data/迅雷下载/1431_part1_tmp/1431_part1/images"
    target_dir = "/data/迅雷下载/1431_part1_tmp/1431_part1/images_converted"
    convert_labelme_files(source_dir, target_dir)
    
    # 方法2: 批量处理多个文件夹
    # source_dirs = [
    #     "/path/to/folder1",
    #     "/path/to/folder2",
    #     "/path/to/folder3"
    # ]
    # target_base_dir = "/path/to/target/base"
    # batch_convert_folders(source_dirs, target_base_dir)