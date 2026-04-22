import os
import sys
import shutil

def filter_yolo_dataset_to_new_dir(images_src, labels_src, labels_dst, keep_classes, images_dst=None, reencode=True):
    """
    将过滤后的YOLO标注写入新文件夹，并可选择同步过滤图像，同时可对保留的类别进行重编码。

    参数:
        images_src (str): 原始图像文件夹路径
        labels_src (str): 原始标注文件夹路径
        labels_dst (str): 新标注文件夹路径（将被创建，如果不存在）
        keep_classes (list[int]): 要保留的类别ID列表
        images_dst (str, optional): 若提供，则复制有有效标注的图像到此文件夹
        reencode (bool): 是否对保留的类别进行重编码（从0开始连续编号），默认为True
    """
    # 创建目标标注文件夹
    os.makedirs(labels_dst, exist_ok=True)
    if images_dst:
        os.makedirs(images_dst, exist_ok=True)

    # 获取所有原始标注文件
    label_files = [f for f in os.listdir(labels_src) if f.endswith('.txt')]
    if not label_files:
        print("未找到任何标注文件。")
        return

    # 准备类别重编码映射
    if reencode:
        # 对保留的类别进行排序（升序），然后重新编码为0,1,2,...
        sorted_keep = sorted(keep_classes)
        class_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_keep)}
        print("类别重编码映射:")
        for old, new in class_mapping.items():
            print(f"  原类别 {old} -> 新类别 {new}")
    else:
        class_mapping = None

    keep_set = set(keep_classes)
    processed_count = 0
    copied_image_count = 0

    for label_file in label_files:
        src_label_path = os.path.join(labels_src, label_file)
        with open(src_label_path, 'r') as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in keep_set:
                # 如果需要重编码，替换类别ID
                if reencode:
                    parts[0] = str(class_mapping[cls_id])
                    filtered_lines.append(' '.join(parts) + '\n')
                else:
                    filtered_lines.append(line)

        # 如果过滤后没有有效标注，跳过该文件（不生成标注，也不复制图像）
        if not filtered_lines:
            continue

        # 写入新标注文件
        dst_label_path = os.path.join(labels_dst, label_file)
        with open(dst_label_path, 'w') as f:
            f.writelines(filtered_lines)
        processed_count += 1
        print(f"已生成标注文件: {dst_label_path}，保留 {len(filtered_lines)} 条标注")

        # 如果需要复制图像，查找并复制对应图像
        if images_dst:
            base_name = os.path.splitext(label_file)[0]
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            found = False
            for ext in image_extensions:
                src_img = os.path.join(images_src, base_name + ext)
                if os.path.exists(src_img):
                    dst_img = os.path.join(images_dst, base_name + ext)
                    shutil.copy2(src_img, dst_img)
                    print(f"已复制图像: {dst_img}")
                    copied_image_count += 1
                    found = True
                    break
            if not found:
                print(f"警告: 未找到对应图像文件 {base_name}.*，跳过复制")

    print(f"处理完成。共生成 {processed_count} 个标注文件。")
    if images_dst:
        print(f"共复制 {copied_image_count} 个图像文件。")
    
    # 可选：生成新的类别映射文件
    if reencode and processed_count > 0:
        mapping_file = os.path.join(labels_dst, "class_mapping.txt")
        with open(mapping_file, 'w') as f:
            f.write("# 原类别ID -> 新类别ID\n")
            for old, new in sorted(class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{old} -> {new}\n")
        print(f"已保存类别映射文件: {mapping_file}")

if __name__ == "__main__":
    # 用户配置区域
    images_src = "D:/迅雷下载/test2/obj_train_data/images"          # 原始图像文件夹
    labels_src = "D:/迅雷下载/test2/obj_train_data/labels"          # 原始标注文件夹
    labels_dst = "D:/迅雷下载/test2/obj_train_data/labels_filtered" # 新标注文件夹（只存放过滤后的标注）
    keep_classes = [0, 1, 2, 5, 10, 13, 14, 32, 33, 34, 37]        # 要保留的类别ID

    # 可选：同步复制图像到新文件夹，只保留有有效标注的图像
    images_dst = "D:/迅雷下载/test2/obj_train_data/images_filtered"  # 设为 None 则不复制图像

    # 是否对保留的类别进行重编码（连续编号）
    reencode = True

    # 检查原始目录是否存在
    if not os.path.isdir(images_src):
        print(f"错误: 原始图像目录不存在 - {images_src}")
        sys.exit(1)
    if not os.path.isdir(labels_src):
        print(f"错误: 原始标注目录不存在 - {labels_src}")
        sys.exit(1)

    filter_yolo_dataset_to_new_dir(
        images_src, labels_src, labels_dst, keep_classes,
        images_dst=images_dst, reencode=reencode
    )