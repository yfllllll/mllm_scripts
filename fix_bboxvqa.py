import json
import os
import glob
from pathlib import Path

def modify_bboxvqa_answers(json_data):
    """
    修改bboxvqa中的answer格式
    
    Args:
        json_data: labelme格式的JSON数据
        
    Returns:
        修改后的JSON数据
    """
    # 遍历所有的shape
    for shape in json_data.get('shapes', []):
        # 确保shape有bboxvqa字段
        if 'bboxvqa' in shape:
            label = shape.get('label', '')
            
            # 遍历每个问答对
            for qa_pair in shape['bboxvqa']:
                answer = qa_pair.get('answer', '')
                question = qa_pair.get('question', '')
                
                # 检查原回答格式并修改
                if answer.startswith('是的'):
                    # 格式: yes, 类别是xxx.
                    qa_pair['answer'] = f"yes, 类别是{label}."
                elif answer.startswith('不是的'):
                    # 格式: no, 类别是xxx.
                    qa_pair['answer'] = f"no, 类别是{label}."
                else:
                    # 如果是其他格式，保持原样或进行相应处理
                    # 这里可以根据需要添加更多处理逻辑
                    pass
    
    return json_data

def process_single_file(json_path, output_dir=None):
    """
    处理单个JSON文件
    
    Args:
        json_path: JSON文件路径
        output_dir: 输出目录（None表示覆盖原文件）
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 修改数据
        modified_data = modify_bboxvqa_answers(data)
        
        # 确定输出路径
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(json_path))
        else:
            output_path = json_path  # 覆盖原文件
        
        # 保存修改后的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f, ensure_ascii=False, indent=2)
        
        print(f"已处理: {json_path} -> {output_path}")
        
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {e}")

def process_directory(input_dir, output_dir=None):
    """
    处理目录下的所有JSON文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（None表示覆盖原文件）
    """
    # 查找所有JSON文件
    json_pattern = os.path.join(input_dir, '*.json')
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"在 {input_dir} 目录中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 处理每个文件
    for json_file in json_files:
        process_single_file(json_file, output_dir)
    
    print(f"处理完成！共处理了 {len(json_files)} 个文件")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='修改labelme数据集中bboxvqa的answer格式')
    parser.add_argument('--input', '-i', default='', 
                       help='输入JSON文件或目录路径')
    parser.add_argument('--output', '-o', default=None,
                       help='输出目录路径（默认覆盖原文件）')

    args = parser.parse_args()


    args = parser.parse_args()
    
    # 检查输入路径
    input_path = args.input

    if args.output:
        output_path = args.output
    else:
        output_path = args.input + "_modified"

    if os.path.isfile(input_path):
        # 单个文件
        process_single_file(input_path, output_path)
    elif os.path.isdir(input_path):
        # 目录
        process_directory(input_path, output_path)
    else:
        print(f"错误: 路径 {input_path} 不存在")    

if __name__ == "__main__":
    main()