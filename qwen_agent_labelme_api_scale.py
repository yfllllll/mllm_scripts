import os
import json
import random
import numpy as np
import requests
import time
import csv
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import shutil
import io  # 新增用于内存字节流

# 后端 API 地址（GroundingDINO 服务）
API_URL = "http://localhost:9020/chat"

def rescale_boxes_in_json(json_content, orig_w, orig_h, scale):
    """将 JSON 中的 bbox 坐标从缩放后尺寸还原到原始尺寸"""
    if scale == 1.0:
        return json_content

    for shape in json_content.get('shapes', []):
        points = shape.get('points', [])
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            # 还原坐标
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            # 确保顺序（左上角 < 右下角）
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            points[0] = [x1, y1]
            points[1] = [x2, y2]

    # 更新图像尺寸为原始尺寸
    json_content['imageWidth'] = orig_w
    json_content['imageHeight'] = orig_h
    return json_content

def save_labelme_format_from_json(json_content, output_path, image_path, image_width, image_height):
    """
    保存从 json_url 下载的 JSON 为 LabelMe 格式，补全必要字段。
    """
    # 确保必要字段存在
    if "imagePath" not in json_content:
        json_content["imagePath"] = os.path.basename(image_path)
    if "imageHeight" not in json_content:
        json_content["imageHeight"] = image_height
    if "imageWidth" not in json_content:
        json_content["imageWidth"] = image_width
    if "imageData" not in json_content:
        json_content["imageData"] = None
    if "description" not in json_content:
        json_content["description"] = ""

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4, ensure_ascii=False)

def process_image(image_path, class_names, output_dir, api_url, service_name, timing_csv_path):
    """
    调用 API，从 json_url 下载 LabelMe JSON 并保存，记录耗时。
    支持图像等比缩放至最大边 1280 以减少显存占用，并在下载后还原坐标。
    """
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return

    # 加载原始图像并获取尺寸
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    max_edge = 1280
    scale = 1.0
    if max(orig_w, orig_h) > max_edge:
        scale = max_edge / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image_resized = image.resize((new_w, new_h), Image.LANCZOS)
        # 将缩放后的图像保存为字节流
        img_buffer = io.BytesIO()
        image_resized.save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()
    else:
        # 不需要缩放，直接用原始图像
        with open(image_path, 'rb') as f:
            img_bytes = f.read()

    # 构建提示文本
    if class_names:
        if isinstance(class_names, dict):
            class_list_str = "，".join(class_names.values())
        else:
            class_list_str = "，".join(class_names)
        prompt_text = f"请检测以下类别：{class_list_str}, 并在调用draw_detections工具时，将存在的矩形框类别名称准确映射为{class_list_str}，我要基于gt计算mAP，评估你的能力。"
    else:
        prompt_text = "请检测图像中的所有目标，并以 JSON 格式报告边界框坐标。"

    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {'message': prompt_text}

    # 记录开始时间
    start_time = time.time()
    try:
        resp = requests.post(api_url, files=files, data=data, timeout=120)
        elapsed = time.time() - start_time
        # 写入耗时到 CSV
        with open(timing_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_path, service_name, elapsed])
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return

    # 处理响应
    try:
        result = resp.json()
        print(f"API response for {os.path.basename(image_path)}: {result}")
    except Exception as e:
        print(f"Error parsing JSON response for {image_path}: {e}")
        # 保存空文件，表示处理失败
        open(output_file, 'w').close()
        return

    # 获取 json_url
    json_url = result.get('json_url')
    if not json_url:
        print(f"No json_url in response for {image_path}. Saving empty file.")
        open(output_file, 'w').close()
        return

    # 下载 LabelMe JSON
    try:
        json_resp = requests.get(json_url, timeout=60)
        json_resp.raise_for_status()
        json_content = json_resp.json()

        # 如果进行了缩放，需要还原坐标
        if scale != 1.0:
            json_content = rescale_boxes_in_json(json_content, orig_w, orig_h, scale)

        # 保存 JSON（确保字段完整）
        save_labelme_format_from_json(json_content, output_file, image_path, orig_w, orig_h)

        print(f"Saved LabelMe JSON from URL: {output_file}")

        # 复制原始图像到输出目录（只复制一次）
        image_filename = os.path.basename(image_path)
        dest_image_path = os.path.join(output_dir, image_filename)
        if not os.path.exists(dest_image_path):
            shutil.copy2(image_path, dest_image_path)
            print(f"Copied image to {dest_image_path}")
    except Exception as e:
        print(f"Error downloading or saving JSON from {json_url} for {image_path}: {e}")
        # 保存空文件，表示处理失败
        open(output_file, 'w').close()

def process_folder(root_dir, folder, api_url,
                   selection_param=1.0, seed=None, dataset_split="all",
                   class_names=None, output_subdir=None, service_name=None, timing_csv_path=None):
    """处理单个文件夹"""
    input_folder = os.path.join(root_dir, folder)
    images_folder = os.path.join(input_folder, 'images')
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    # 确定输出目录
    if output_subdir:
        output_dir = os.path.join(input_folder, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = images_folder

    # 获取图片列表
    split_folders = {'train': 'train.jsonl', 'val': 'val.jsonl', 'test': 'test.jsonl'}
    split_file = split_folders.get(dataset_split, None)

    if dataset_split != "all":
        if split_file and os.path.exists(os.path.join(input_folder, split_file)):
            print(f"Processing {dataset_split} dataset...")
            with open(os.path.join(input_folder, split_file), 'r') as jsonl_file:
                lines = jsonl_file.readlines()
            image_files = [image for line in lines for image in json.loads(line.strip())["images"]]
            if image_files and not os.path.isabs(image_files[0]):
                image_files = [os.path.join(images_folder, f) for f in image_files]
        else:
            print(f"No {dataset_split}.jsonl file found, skipping {dataset_split} dataset.")
            return
    else:
        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 确定处理数量
    if isinstance(selection_param, float) and 0 < selection_param <= 1:
        image_count = int(len(image_files) * selection_param)
    elif isinstance(selection_param, int) and selection_param > 1:
        image_count = min(selection_param, len(image_files))
    else:
        raise ValueError("selection_param must be a float between 0 and 1 (percentage) or an integer greater than 1 (count)")

    selected_images = random.sample(image_files, image_count)

    for image_path in tqdm(selected_images, desc=f"Processing {dataset_split} dataset", unit="image"):
        try:
            process_image(image_path, class_names, output_dir, api_url, service_name, timing_csv_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    print(f"All JSON files saved in: {output_dir}")

def process_folders(root_dir, folder_list, api_url,
                    selection_param=1.0, seed=None, dataset_split="all",
                    class_names=None, output_subdir=None, service_name=None, timing_csv_path=None):
    """处理多个文件夹"""
    for folder in folder_list:
        print(f"Processing folder: {folder}")
        try:
            process_folder(root_dir, folder, api_url,
                           selection_param=selection_param, seed=seed,
                           dataset_split=dataset_split, class_names=class_names,
                           output_subdir=output_subdir, service_name=service_name,
                           timing_csv_path=timing_csv_path)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

if __name__ == "__main__":
    # ==================== 配置参数 ====================
    root_dir = "/mnt/disk/lyf/datasets/motuo_feijidong"
    folder_list = ["motuo_feijidong"]
    dataset_split = "all"
    selection_param = 0.1
    seed = 52
    classname_list = ["自行车", "电瓶车", "摩托车"]

    # 服务配置
    API_URL = "http://localhost:9019/chat"
    SERVICE_NAME = "groundingdino"
    output_subdir = "pred_service1_scale"          # 输出子目录
    timing_csv = "./timing_service1.csv"     # 计时结果文件

    # 确保计时文件目录存在，并写入表头
    os.makedirs(os.path.dirname(timing_csv) if os.path.dirname(timing_csv) else '.', exist_ok=True)
    if not os.path.exists(timing_csv):
        with open(timing_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image_path", "service", "elapsed_seconds"])

    process_folders(root_dir, folder_list,
                    api_url=API_URL,
                    selection_param=selection_param,
                    seed=seed,
                    dataset_split=dataset_split,
                    class_names=classname_list,
                    output_subdir=output_subdir,
                    service_name=SERVICE_NAME,
                    timing_csv_path=timing_csv)
    print('Processing completed.')