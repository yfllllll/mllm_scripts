import json
import os

def extract_classes(json_dir, output_file):
    classes = set()
    
    # 遍历目录下所有文件
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 提取每个 shape 的标签
                    for shape in data.get('shapes', []):
                        classes.add(shape['label'])
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")

    # 将类别排序并写入文件
    sorted_classes = sorted(list(classes))
    with open(output_file, 'w', encoding='utf-8') as f:
        for cls in sorted_classes:
            f.write(f"{cls}\n")
    
    print(f"统计完成！共发现 {len(sorted_classes)} 个类别。")
    print(f"类别列表已保存至: {output_file}")

# 使用示例
# 请将 'your_json_folder' 替换为你存放 JSON 文件的文件夹路径
extract_classes('/mnt/disk/lyf/datasets/coco/obj_vqafine', 'coco_classes.txt')
