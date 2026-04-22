import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import torch
import json
import re
import yaml
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams

# YOLO 格式保存函数
def save_yolo_format(predictions, output_path, class_id, image_width, image_height, append=False):
    mode = 'a' if append else 'w'  # 'a' for append, 'w' for overwrite
    with open(output_path, mode) as f:
        for box in predictions:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2.0 / image_width
            y_center = (y1 + y2) / 2.0 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
# 初始化推理引擎函数
def initialize_engine_quant(engine_type='pt', model_id_or_path=None):
    if engine_type == 'pt':
        # 初始化参数字典
        pt_engine_kwargs = {}
        
        # 检查 model_id_or_path 是否是一个本地路径
        
        if os.path.isdir(model_id_or_path):
            args_json_path = os.path.join(model_id_or_path, 'args.json')
            # quantize_config_path = os.path.join(model_id_or_path, 'quantize_config.json')
            if os.path.exists(args_json_path):
                # 尝试读取 args.json 文件
                try:
                    with open(args_json_path, 'r', encoding='utf-8') as f:
                        args = json.load(f)
                        # 获取 model_type 字段并加入到参数字典
                        if 'model_type' in args:
                            pt_engine_kwargs['model_type'] = args['model_type']
                        else:
                            print(f"Warning: 'model_type' field not found in {args_json_path}")
                except Exception as e:
                    print(f"Error reading {args_json_path}: {e}")
            else:
                print(f"Warning: args.json not found in {model_id_or_path}")
            # if os.path.exists(quantize_config_path):
            #     # 尝试读取 args.json 文件
            #     try:
            #         with open(quantize_config_path, 'r', encoding='utf-8') as f:
            #             args = json.load(f)
            #             # 获取 model_type 字段并加入到参数字典
            #             pt_engine_kwargs['quantization_config'] = args
            #     except Exception as e:
            #         print(f"Error reading {quantize_config_path}: {e}")
            # else:
            #     print(f"Warning: args.json not found in {model_id_or_path}")
        # 添加其他 PtEngine 参数
        pt_engine_kwargs['max_batch_size'] = 8

        pt_engine_kwargs['torch_dtype'] =torch.float16

        # 初始化 PtEngine
        print(pt_engine_kwargs)
        engine = PtEngine(model_id_or_path, **pt_engine_kwargs)

    elif engine_type == 'vllm':
        engine = VllmEngine(model_id_or_path, max_model_len=32768, limit_mm_per_prompt={'image': 5, 'video': 2})
    elif engine_type == 'lmdeploy':
        engine = LmdeployEngine(model_id_or_path, vision_batch_size=8)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")
    
    return engine


# 解析 response 中的 bbox 和类别
def parse_response_boxes(response):
    """
    从响应文本中解析检测框坐标并按比例还原，并且支持多个类别。
    """
    # 创建一个字典，键是类别，值是框

    # 正则表达式修改为支持多类别的解析，输出格式：{ "bbox_2d": [x1, y1, x2, y2], "label": "{category_name}" }
    box_pattern = r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"(?:,\s*"sub lable":\s*"(.*?)")?'
    #r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"'

    matches = re.findall(box_pattern, response[0])
    '''potential_lines = []
    for line in response[0].split('\n'):
        if '{"bbox_2d":' in line:
            potential_lines.append(line)

    # 调整正则表达式，使其更具包容性
    box_pattern = r'\{"bbox_2d":\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"([^"]+)"\}'

    matches = []
    for line in potential_lines:
        match = re.search(box_pattern, line)
        if match:
            matches.append(match.groups())'''
    # 解析每一个框
    category_boxes = {}
    for match in matches:
        x1, y1, x2, y2, class_name,sub_classname = match
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = x1 
        y1 = y1
        x2 = x2 
        y2 = y2 
        # 检查类别是否已经在字典中，如果不在则创建一个空列表
        if sub_classname:
            class_name = class_name + '/' + sub_classname
        if class_name not in category_boxes:
            category_boxes[class_name] = []
        # 如果类别在指定的类别名列表中，则将框加入对应的类别
        category_boxes[class_name].append([x1, y1, x2, y2])

    # 生成格式化后的检测框列表
    boxes = []
    for category, box_list in category_boxes.items():
        for box in box_list:
            boxes.append((category, box))
    return boxes


# 处理每张图片
def process_image(image_path, class_names, output_dir, model):
    # 检查该图片是否已经处理过
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return
    
    # 加载图像
    data = Image.open(image_path).convert("RGB")
    image_width, image_height = data.size
    # 构建 query，一次性询问所有类别
    class_list_str = "，".join(class_names.values())
    category_name = f"请检测图像中包含的{class_list_str}等目标，并以坐标形式返回它们的位置，如果不存在则不用输出坐标，输出格式应为:{{\"bbox_2d\": [x1,y1,x2,y2],\"label\":\"XXX\"}}，其中XXX为目标名称。"
    
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=4096)
    placeholder = "<|image_pad|>"
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{category_name}<|im_end|>\n"
              "<|im_start|>assistant\n")
    # 构建推理请求
    inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                'image': data
            },
        }
    
    # 模型推理
    outputs = model.generate([inputs], sampling_params)
    output_text = [output.outputs[0].text for output in outputs]
    print(output_text)

    # 解析检测结果
    parsed_boxes = parse_response_boxes(output_text)
    
    if not parsed_boxes:
        # 如果没有检测框，生成一个空文件
        open(output_file, 'w').close()
        print(f"No detections for {image_path}. Created empty file: {output_file}")
        return
    
    for class_name, box in parsed_boxes:
        # 检查类别是否在 class_names 中，防止出现错误
        if class_name not in class_names.values():
            print(f"Skipping unknown class: {class_name}")
            continue

        class_id = list(class_names.values()).index(class_name)  # 获取类别对应的 ID
        save_yolo_format([box], output_file, class_id, image_width, image_height, append=True)

    print(f"Processed and saved: {image_path}")



# 修改后的处理单个文件夹函数，支持选择处理train, val, test
def process_folder(root_dir, folder, saved_folder='pred', model=None, selection_param=1.0, seed=None, dataset_split="all"):
    input_folder = os.path.join(root_dir, folder)
    
    yaml_files = ['data.yaml', 'dataset.yaml']
    yaml_path = next((os.path.join(input_folder, f) for f in yaml_files if os.path.exists(os.path.join(input_folder, f))), None)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found in {input_folder}")
    
    with open(yaml_path, 'r') as yaml_file:
        data_info = yaml.safe_load(yaml_file)
    
    class_names = data_info['names']  # 类别名称
    images_folder = os.path.join(input_folder, 'images')
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    # 根据 dataset_split 来选择处理哪些数据集
    split_folders = {'train': 'train.jsonl', 'val': 'val.jsonl', 'test': 'test.jsonl'}
    
    # 根据 dataset_split 参数决定要处理的数据
    split_file = split_folders.get(dataset_split, None)
    
    if dataset_split != "all":
        # 如果选择了 train, val 或 test，检查是否有对应的 jsonl 文件
        if split_file and os.path.exists(os.path.join(input_folder, split_file)):
            print(f"Processing {dataset_split} dataset...")
            with open(os.path.join(input_folder, split_file), 'r') as jsonl_file:
                lines = jsonl_file.readlines()
            
            #image_files = [json.loads(line.strip())["images"] for line in lines]
            image_files = [image for line in lines for image in json.loads(line.strip())["images"]]
        else:
            print(f"No {dataset_split}.jsonl file found, skipping {dataset_split} dataset.")
            return
    else:
        # 如果是处理所有数据集
        # 获取所有图片文件
        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 处理随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 确定处理的图片数量
    if isinstance(selection_param, float) and 0 < selection_param <= 1:
        # 如果是浮点数且在 (0, 1] 之间，按百分比处理
        image_count = int(len(image_files) * selection_param)
    elif isinstance(selection_param, int) and selection_param > 1:
        # 如果是整数且大于 1，按影像个数处理
        image_count = min(selection_param, len(image_files))
    else:
        raise ValueError("selection_param must be a float between 0 and 1 (percentage) or an integer greater than 1 (count)")

    selected_images = random.sample(image_files, image_count)
    
    # 保存结果的文件夹
    pred_folder = os.path.join(input_folder, saved_folder, dataset_split)
    os.makedirs(pred_folder, exist_ok=True)

    # 逐个处理选中的图像
    for image_path in tqdm(selected_images, desc=f"Processing {dataset_split} dataset", unit="image"):
        process_image(image_path, class_names, pred_folder, model)

    print(f"All predictions saved in: {pred_folder}")


# 修改后的处理文件夹列表函数，支持灵活设置百分比或影像个数
def process_folders(root_dir, folder_list, saved_folder='pred', model=None, selection_param=1.0, seed=None, dataset_split="all"):
    for i, folder in enumerate(folder_list):
        print(f"Processing folder: {folder}")

        try:
            process_folder(root_dir, folder, saved_folder=saved_folder, model=model,
                           selection_param=selection_param, seed=seed, dataset_split=dataset_split)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")


# 主程序入口
if __name__ == "__main__":
    
    
    # 设置路径
    root_dir = "/data1/lyf/formate_t/coco_format"
    folder_list = ["zixingche"]
    saved_folder = 'pred_multi_engine_only_qwen25'
    
    # 选择处理哪些数据集: "train", "val", "test", 或 "all"
    dataset_split = "val"  # 可以设置为 "train", "val", "test", 或 "all"
    
    selection_param = 1.  # 对第一个数据集处理50%的图片，第二个数据集处理100张图片
    seed = 42 
    
    # 开始处理
    model_name = "/data3/models/Qwen2.5-VL-72B-Instruct"

    llm = LLM(
        model=model_name,
        #max_model_len=4096,
        tensor_parallel_size=4,
        #gpu_memory_utilization=0.15,
        #dtype="bfloat16",
       # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
         #    "max_pixels": 1280 * 28 * 28,
          #   "fps": 1,
        #}, 
    )
    process_folders(root_dir, folder_list, saved_folder=saved_folder, model=llm, selection_param=selection_param, seed=seed, dataset_split=dataset_split)
    
    print('Processing completed.')
