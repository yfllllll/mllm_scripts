import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import json
import random
import numpy as np
import base64
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from io import BytesIO
import re
import shutil
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 准备vLLM输入的函数（适配Qwen3-VL）
def prepare_inputs_for_vllm(messages, processor):
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

# 解析JSON输出
def parse_json(json_output):
    """解析JSON输出"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

# 修改后的保存为LabelMe格式的函数
def save_labelme_format(predictions, output_text, output_path, image_path, image_width, image_height):
    shapes = []
    for box in predictions:
        class_name, coordinates = box
        x1, y1, x2, y2 = coordinates

        # 检查坐标合法性
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid coordinates for {class_name}: {coordinates}. Skipping.")
            continue

        # 使用矩形形状类型
        points = [
            [x1, y1],  
            [x2, y2],
        ]
        
        shape = {
            "label": class_name,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",  # 修改为矩形
            "flags": {}
        }
        shapes.append(shape)

    # 将图像转换为 Base64 编码
    image_data = None
    '''with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")'''

    annotation = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),  # Image file name
        "imageData": image_data,  # Base64 image data
        "imageHeight": image_height,
        "imageWidth": image_width,
        "description": output_text
    }

    # with open(output_path, 'w') as f:
    #     json.dump(annotation, f, indent=4)
    with open(output_path, 'w', encoding='utf-8') as f:  # 添加encoding='utf-8'
        json.dump(annotation, f, indent=4, ensure_ascii=False)  # 添加ensure_ascii=False

# 解析 response 中的 bbox 和类别，将归一化坐标转换为绝对坐标
def parse_response_boxes(response, image_width, image_height):
    """
    从响应文本中解析检测框坐标，将Qwen3-VL的归一化坐标转换为绝对坐标
    Qwen3-VL使用归一化坐标（0-1000），需要转换为绝对像素坐标
    """
    # 首先尝试解析JSON格式
    try:
        # 清理响应文本，提取JSON部分
        cleaned_response = parse_json(response[0])
        data = json.loads(cleaned_response)
        
        boxes = []
        if isinstance(data, list):
            for item in data:
                if "bbox_2d" in item and "label" in item:
                    # Qwen3-VL使用归一化坐标（0-1000），需要转换为绝对坐标
                    x1_norm, y1_norm, x2_norm, y2_norm = item["bbox_2d"]
                    
                    # 将归一化坐标转换为绝对坐标
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
            # 处理单个检测框的情况
            x1_norm, y1_norm, x2_norm, y2_norm = data["bbox_2d"]
            
            # 将归一化坐标转换为绝对坐标
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
        # 如果JSON解析失败，回退到正则表达式
        box_pattern = r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"(?:,\s*"sub lable":\s*"(.*?)")?'
        matches = re.findall(box_pattern, response[0])

        category_boxes = {}
        for match in matches:
            x1_norm, y1_norm, x2_norm, y2_norm, class_name, sub_classname = match
            x1_norm, y1_norm, x2_norm, y2_norm = map(int, [x1_norm, y1_norm, x2_norm, y2_norm])
            
            # 将归一化坐标转换为绝对坐标
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

# 处理每张图片
def process_image(image_path, class_names, output_dir, model, processor):
    # 检查该图片是否已经处理过
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    
    # 构建消息（适配Qwen3-VL格式）
    if class_names:
        class_list_str = "，".join(class_names.values())
        prompt_text = f"请检测图像中包含的{class_list_str}等目标，以JSON格式报告边界框坐标,其中label以类似：label|brife described instance这种形式给出"
    else:
        prompt_text = f"先描述一下这个图像，然后基于你的描述检测图像中的目标（20个目标即可）,以 JSON 格式报告边界框坐标,其中label以类似：label/brife described instance这种形式给出"
    
    # 构建消息格式（Qwen3-VL格式）
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # 准备输入
    inputs = prepare_inputs_for_vllm(messages, processor)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
        top_k=-1,
        stop_token_ids=[],
    )
    
    # 模型推理
    outputs = model.generate([inputs], sampling_params)
    output_text = [output.outputs[0].text for output in outputs]
    print(f"Model output: {output_text}")

    # 解析检测结果，传入图像尺寸以进行坐标转换
    parsed_boxes = parse_response_boxes(output_text, image_width, image_height)
    
    if not parsed_boxes:
        # 如果没有检测框，生成一个空文件
        open(output_file, 'w').close()
        print(f"No detections for {image_path}. Created empty file: {output_file}")
        return
    
    # 将 LabelMe 格式保存到文件
    save_labelme_format(parsed_boxes, output_text, output_file, image_path, image_width, image_height)
    print(f"Processed and saved: {image_path}")

# 修改后的处理文件夹函数，支持选择处理train, val, test
def process_folder(root_dir, folder, saved_folder='pred', model=None, processor=None, selection_param=1.0, seed=None, dataset_split="all"):
    input_folder = os.path.join(root_dir, folder)
    class_names = None
    """如果需要自定义类别，可以打开以下注释"""
    ##########################################
    # yaml_files = ['data.yaml', 'dataset.yaml']
    # yaml_path = next((os.path.join(input_folder, f) for f in yaml_files if os.path.exists(os.path.join(input_folder, f))), None)
    # if not os.path.exists(yaml_path):
    #     raise FileNotFoundError(f"data.yaml not found in {input_folder}")
    
    # with open(yaml_path, 'r') as yaml_file:
    #     data_info = yaml.safe_load(yaml_file)
    
    # class_names = data_info['names']  # 类别名称
    ############################################
    
    images_folder = os.path.join(input_folder, 'images')
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    # 根据 dataset_split 来选择处理哪些数据集
    split_folders = {'train': 'train.jsonl', 'val': 'val.jsonl', 'test': 'test.jsonl'}
    
    # 根据 dataset_split 参数决定要处理的数据
    split_file = split_folders.get(dataset_split, None)
    
    if dataset_split != "all":
        if split_file and os.path.exists(os.path.join(input_folder, split_file)):
            print(f"Processing {dataset_split} dataset...")
            with open(os.path.join(input_folder, split_file), 'r') as jsonl_file:
                lines = jsonl_file.readlines()
            
            image_files = [image for line in lines for image in json.loads(line.strip())["images"]]
        else:
            print(f"No {dataset_split}.jsonl file found, skipping {dataset_split} dataset.")
            return
    else:
        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 处理随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 确定处理的图片数量
    if isinstance(selection_param, float) and 0 < selection_param <= 1:
        image_count = int(len(image_files) * selection_param)
    elif isinstance(selection_param, int) and selection_param > 1:
        image_count = min(selection_param, len(image_files))
    else:
        raise ValueError("selection_param must be a float between 0 and 1 (percentage) or an integer greater than 1 (count)")

    
    selected_images = random.sample(image_files, image_count)
    
    # 保存结果的文件夹
    pred_folder = os.path.join(input_folder, saved_folder, dataset_split)
    os.makedirs(pred_folder, exist_ok=True)

    # 逐个处理选中的图像
    for image_path in tqdm(selected_images, desc=f"Processing {dataset_split} dataset", unit="image"):
        try:
            process_image(image_path, class_names, pred_folder, model, processor)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    print(f"All predictions saved in: {pred_folder}")

# 修改后的处理文件夹列表函数，支持灵活设置百分比或影像个数
def process_folders(root_dir, folder_list, saved_folder='pred', model=None, processor=None, selection_param=1.0, seed=None, dataset_split="all"):
    for i, folder in enumerate(folder_list):
        print(f"Processing folder: {folder}")

        try:
            process_folder(root_dir, folder, saved_folder=saved_folder, model=model, processor=processor,
                           selection_param=selection_param, seed=seed, dataset_split=dataset_split)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

# 主程序入口
if __name__ == "__main__":
    root_dir = "/mnt/data/lyf/ziwei/mulobj_part13_20251101_20251103"
    folder_list = ["1431_part1","1431_part2"]
    saved_folder = 'labelme'

    dataset_split = "all"  # 可以设置为 "train", "val", "test", 或 "all"
    selection_param = 1.0
    seed = 52 #42 

    # 使用Qwen3-VL模型
    model_name = "/mnt/data/lyf/Qwen3-VL-32B-Instruct"  # 或者您使用的Qwen3-VL模型路径
    
    # 初始化processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 初始化vLLM模型
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        enforce_eager=False,
        max_model_len=41960,
        tensor_parallel_size=4,  # 根据您的GPU数量调整
        seed=0,
        dtype="bfloat16",
    )

    process_folders(root_dir, folder_list, saved_folder=saved_folder, model=llm, processor=processor, 
                   selection_param=selection_param, seed=seed, dataset_split=dataset_split)
    print('Processing completed.')