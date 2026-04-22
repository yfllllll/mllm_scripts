import os
import json
import random
import numpy as np
import base64
import time
import csv
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import re
import shutil
# 将 PIL 图像转换为 base64 字符串（统一为 JPEG 格式）
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

# 解析 JSON 输出
def parse_json(json_output):
    """解析JSON输出"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

# 保存为 LabelMe 格式
def save_labelme_format(predictions, output_text, output_path, image_path, image_width, image_height):
    shapes = []
    for box in predictions:
        class_name, coordinates = box
        x1, y1, x2, y2 = coordinates

        if x1 >= x2 or y1 >= y2:
            print(f"Invalid coordinates for {class_name}: {coordinates}. Skipping.")
            continue

        points = [[x1, y1], [x2, y2]]
        shape = {
            "label": class_name,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(shape)

    annotation = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "description": output_text
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=4, ensure_ascii=False)

# 解析 response 中的 bbox 和类别
def parse_response_boxes(response, image_width, image_height):
    """
    从响应文本中解析检测框坐标，将Qwen3-VL的归一化坐标转换为绝对坐标
    Qwen3-VL使用归一化坐标（0-1000），需要转换为绝对像素坐标
    """
    try:
        cleaned_response = parse_json(response)
        data = json.loads(cleaned_response)

        boxes = []
        if isinstance(data, list):
            for item in data:
                if "bbox_2d" in item and "label" in item:
                    x1_norm, y1_norm, x2_norm, y2_norm = item["bbox_2d"]
                    x1 = int(x1_norm / 1000 * image_width)
                    y1 = int(y1_norm / 1000 * image_height)
                    x2 = int(x2_norm / 1000 * image_width)
                    y2 = int(y2_norm / 1000 * image_height)

                    class_name = item["label"]
                    sub_classname = item.get("sub lable", "")
                    if sub_classname:
                        class_name = class_name + '/' + sub_classname
                    boxes.append((class_name, [x1, y1, x2, y2]))
        elif isinstance(data, dict) and "bbox_2d" in data:
            x1_norm, y1_norm, x2_norm, y2_norm = data["bbox_2d"]
            x1 = int(x1_norm / 1000 * image_width)
            y1 = int(y1_norm / 1000 * image_height)
            x2 = int(x2_norm / 1000 * image_width)
            y2 = int(y2_norm / 1000 * image_height)

            class_name = data["label"]
            sub_classname = data.get("sub lable", "")
            if sub_classname:
                class_name = class_name + '/' + sub_classname
            boxes.append((class_name, [x1, y1, x2, y2]))

        return boxes
    except Exception as e:
        print(f"JSON parsing failed: {e}, falling back to regex")
        # 正则表达式回退
        box_pattern = r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"(?:,\s*"sub lable":\s*"(.*?)")?'
        matches = re.findall(box_pattern, response)

        category_boxes = {}
        for match in matches:
            x1_norm, y1_norm, x2_norm, y2_norm, class_name, sub_classname = match
            x1_norm, y1_norm, x2_norm, y2_norm = map(int, [x1_norm, y1_norm, x2_norm, y2_norm])

            x1 = int(x1_norm / 1000 * image_width)
            y1 = int(y1_norm / 1000 * image_height)
            x2 = int(x2_norm / 1000 * image_width)
            y2 = int(y2_norm / 1000 * image_height)

            if sub_classname:
                class_name = class_name + '/' + sub_classname
            if class_name not in category_boxes:
                category_boxes[class_name] = []
            category_boxes[class_name].append([x1, y1, x2, y2])

        boxes = []
        for category, box_list in category_boxes.items():
            for box in box_list:
                boxes.append((category, box))
        return boxes

# 处理每张图片（通过 API 调用）
def process_image(image_path, class_names, output_dir, api_url, model_name, service_name, timing_csv_path):
    # 输出文件与图像文件在同一目录，文件名相同但扩展名为 .json
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return

    # 加载图像并获取尺寸
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    # 构建提示文本（兼容字典和列表）
    if class_names:
        if isinstance(class_names, dict):
            class_list_str = "，".join(class_names.values())
        else:
            # 假设是列表
            class_list_str = "，".join(class_names)
        prompt_text = f"请检测以下类别目标：{class_list_str}，并以JSON格式报告边界框坐标"
    else:
        prompt_text = "先描述一下这个图像，然后基于你的描述检测图像中的目标（20个目标即可）,以 JSON 格式报告边界框坐标,其中label以类似：label/brief described instance这种形式给出"

    # 将图像转为 base64
    base64_image = image_to_base64(image)

    # 构建 OpenAI API 兼容的请求体
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": base64_image}},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1,
    }

    # 计时开始
    start_time = time.time()
    try:
        resp = requests.post(api_url, json=payload, timeout=120)
        elapsed = time.time() - start_time
        # 写入耗时到 CSV
        with open(timing_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_path, service_name, elapsed])
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return

    if resp.status_code != 200:
        print(f"API returned status {resp.status_code}: {resp.text}")
        return

    try:
        response_data = resp.json()
        output_text = response_data['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error parsing response for {image_path}: {e}")
        return

    print(f"Model output: {output_text}")

    # 解析检测结果
    parsed_boxes = parse_response_boxes(output_text, image_width, image_height)

    if not parsed_boxes:
        # 无检测框，保存空文件（表示已处理但无目标）
        open(output_file, 'w').close()
        print(f"No detections for {image_path}. Created empty file: {output_file}")
        return

    # 保存 LabelMe 格式
    save_labelme_format(parsed_boxes, output_text, output_file, image_path, image_width, image_height)
    print(f"Processed and saved: {image_path}")
    image_filename = os.path.basename(image_path)
    dest_image_path = os.path.join(output_dir, image_filename)
    if not os.path.exists(dest_image_path):
        shutil.copy2(image_path, dest_image_path)
        print(f"Copied image to {dest_image_path}")
# 处理文件夹
def process_folder(root_dir, folder, api_url=None, model_name=None,
                   selection_param=1.0, seed=None, dataset_split="all",
                   class_names=None, output_subdir=None, service_name=None, timing_csv_path=None):
    input_folder = os.path.join(root_dir, folder)
    images_folder = os.path.join(input_folder, 'images')
    if output_subdir:
        output_dir = os.path.join(input_folder, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = images_folder

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    # 根据 dataset_split 获取图片列表
    split_folders = {'train': 'train.jsonl', 'val': 'val.jsonl', 'test': 'test.jsonl'}
    split_file = split_folders.get(dataset_split, None)

    if dataset_split != "all":
        if split_file and os.path.exists(os.path.join(input_folder, split_file)):
            print(f"Processing {dataset_split} dataset...")
            with open(os.path.join(input_folder, split_file), 'r') as jsonl_file:
                lines = jsonl_file.readlines()
            # 注意：这里假设 jsonl 每行是 {"images": [...]} 格式
            image_files = [image for line in lines for image in json.loads(line.strip())["images"]]
            # 如果 jsonl 中只存了文件名，需补全路径
            if not os.path.isabs(image_files[0]):
                image_files = [os.path.join(images_folder, f) for f in image_files]
        else:
            print(f"No {dataset_split}.jsonl file found, skipping {dataset_split} dataset.")
            return
    else:
        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 随机种子
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

    # 逐个处理
    for image_path in tqdm(selected_images, desc=f"Processing {dataset_split} dataset", unit="image"):
        try:
            process_image(image_path, class_names, output_dir, api_url, model_name, service_name, timing_csv_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    print(f"All JSON files saved in: {output_dir}")

# 处理文件夹列表
def process_folders(root_dir, folder_list, api_url=None, model_name=None,
                    selection_param=1.0, seed=None, dataset_split="all",
                    class_names=None, output_subdir=None, service_name=None, timing_csv_path=None):
    for folder in folder_list:
        print(f"Processing folder: {folder}")
        try:
            process_folder(root_dir, folder, api_url=api_url,
                           model_name=model_name, selection_param=selection_param,
                           seed=seed, dataset_split=dataset_split, class_names=class_names,
                           output_subdir=output_subdir, service_name=service_name,
                           timing_csv_path=timing_csv_path)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

# 主程序入口
if __name__ == "__main__":
    root_dir = "/mnt/disk/lyf/datasets/"
    folder_list = ["test2"]
    dataset_split = "all"
    selection_param = 0.1
    seed = 52
    classname_list = ["小汽车", "SUV", "面包车", "大巴", "卡车", "厢货", "挖掘机",  "电瓶车", "摩托车", "自行车", "人",]

    # vLLM 服务配置
    api_url = "http://localhost:8002/v1/chat/completions"  # 根据实际端口修改
    model_name = "QwenVl"                                 # 与启动服务时指定的 --served-model-name 一致

    output_subdir = "pred_qwen27_test2"                         # 输出子目录
    service_name = "qwen3vl"                              # 服务标识
    timing_csv = "./timing_service2_test2.csv"                  # 计时结果文件

    # 确保计时文件目录存在，并写入表头
    os.makedirs(os.path.dirname(timing_csv) if os.path.dirname(timing_csv) else '.', exist_ok=True)
    if not os.path.exists(timing_csv):
        with open(timing_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image_path", "service", "elapsed_seconds"])

    process_folders(root_dir, folder_list,
                    api_url=api_url,
                    model_name=model_name,
                    selection_param=selection_param,
                    seed=seed,
                    dataset_split=dataset_split,
                    class_names=classname_list,
                    output_subdir=output_subdir,
                    service_name=service_name,
                    timing_csv_path=timing_csv)
    print('Processing completed.')
