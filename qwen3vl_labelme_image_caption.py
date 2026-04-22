import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
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
import time

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

# 修改后的保存为LabelMe格式的函数（适配图像描述）
def save_labelme_format(descriptions, output_path, image_path, image_width, image_height):
    """保存图像描述到LabelMe格式"""
    # 将图像转换为 Base64 编码（可选，如果需要）
    image_data = None
    '''
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    '''

    annotation = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],  # 空列表，因为我们不标注目标
        "imagePath": os.path.basename(image_path),  # 图像文件名
        "imageData": image_data,  # Base64图像数据（可选）
        "imageHeight": image_height,
        "imageWidth": image_width,
        "descriptions": descriptions,  # 存储5个描述作为列表
        "description_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 添加时间戳
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=4, ensure_ascii=False)

# 处理每张图片，获取5次描述
def process_image(image_path, output_dir, model, processor):
    # 检查该图片是否已经处理过
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    
    descriptions = []  # 存储5次描述
    prompt_text = "请详细描述这张图像的内容，包括场景、物体、颜色、动作、情感等所有你注意到的细节。"
    
    print(f"Processing {image_path}...")
    
    # 进行5次独立的询问
    for i in range(5):
        try:
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
            
            # 设置采样参数（可以使用不同的temperature来增加多样性）
            sampling_params = SamplingParams(
                # temperature=0.7 + i * 0.1,  # 每次稍微调整temperature以获得不同描述
                max_tokens=1024,
                top_k=-1,
                stop_token_ids=[],
            )
            
            # 模型推理
            outputs = model.generate([inputs], sampling_params)
            output_text = outputs[0].outputs[0].text.strip()
            
            # 清理输出文本（移除可能的标记）
            cleaned_text = output_text.replace("```json", "").replace("```", "").strip()
            descriptions.append({
                "id": i + 1,
                "description": cleaned_text,
                "temperature": sampling_params.temperature
            })
            
            print(f"  Description {i+1}: {cleaned_text[:100]}...")
            
       
            
        except Exception as e:
            print(f"  Error getting description {i+1}: {e}")
            # 如果出错，添加一个占位描述
            descriptions.append({
                "id": i + 1,
                "description": f"Error generating description: {str(e)}",
                "temperature": 0.7 + i * 0.1
            })
    
    # 将描述保存到文件
    save_labelme_format(descriptions, output_file, image_path, image_width, image_height)
    print(f"Saved descriptions to: {output_file}")

# 修改后的处理文件夹函数
def process_folder(root_dir, folder, saved_folder='descriptions', model=None, processor=None, 
                   selection_param=1.0, seed=None, dataset_split="all"):
    input_folder = os.path.join(root_dir, folder)
    
    images_folder = os.path.join(input_folder, 'images')
    if not os.path.exists(images_folder):
        # 尝试直接查找图像文件
        images_folder = input_folder
        print(f"Warning: 'images' subfolder not found, using {input_folder} directly")
    
    # 根据 dataset_split 来选择处理哪些数据集
    split_folders = {'train': 'train.jsonl', 'val': 'val.jsonl', 'test': 'test.jsonl'}
    
    # 根据 dataset_split 参数决定要处理的数据
    if dataset_split != "all":
        split_file = split_folders.get(dataset_split)
        if split_file and os.path.exists(os.path.join(input_folder, split_file)):
            print(f"Processing {dataset_split} dataset...")
            with open(os.path.join(input_folder, split_file), 'r') as jsonl_file:
                lines = jsonl_file.readlines()
            
            image_files = []
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    # 处理不同的JSONL格式
                    if "images" in data:
                        image_files.extend(data["images"])
                    elif "image" in data:
                        image_files.append(data["image"])
                except:
                    continue
            
            # 确保路径正确
            image_files = [os.path.join(images_folder, os.path.basename(f)) for f in image_files]
        else:
            print(f"No {dataset_split}.jsonl file found, processing all images in folder.")
            image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    else:
        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
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
    
    # 如果选择所有图片，直接使用所有文件
    if selection_param == 1.0 and isinstance(selection_param, float):
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, image_count)
    
    # 保存结果的文件夹
    pred_folder = os.path.join(input_folder, saved_folder, dataset_split)
    os.makedirs(pred_folder, exist_ok=True)
    
    print(f"Total images found: {len(image_files)}")
    print(f"Selected images to process: {len(selected_images)}")
    print(f"Output directory: {pred_folder}")
    
    # 逐个处理选中的图像
    for image_path in tqdm(selected_images, desc=f"Processing {dataset_split} dataset", unit="image"):
        try:
            if os.path.exists(image_path):
                process_image(image_path, pred_folder, model, processor)
            else:
                print(f"Warning: Image not found: {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"All descriptions saved in: {pred_folder}")

# 处理文件夹列表函数
def process_folders(root_dir, folder_list, saved_folder='descriptions', model=None, processor=None, 
                   selection_param=1.0, seed=None, dataset_split="all"):
    for i, folder in enumerate(folder_list):
        print(f"Processing folder {i+1}/{len(folder_list)}: {folder}")
        print("=" * 50)
        
        try:
            process_folder(root_dir, folder, saved_folder=saved_folder, model=model, 
                          processor=processor, selection_param=selection_param, 
                          seed=seed, dataset_split=dataset_split)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 50)

# 主程序入口
if __name__ == "__main__":
    # 配置参数
    root_dir = "/mnt/data/lyf/datasets"
    folder_list = ["1431_part1"]
    saved_folder = 'descriptions'
    
    dataset_split = "all"  # 可以设置为 "train", "val", "test", 或 "all"
    selection_param = 1.  # 处理10%的图片，可以根据需要调整
    seed = 52
    
    # 使用Qwen3-VL模型
    model_name = "/mnt/data/lyf/Qwen3-VL-32B-Instruct"
    
    print("Initializing processor...")
    # 初始化processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Initializing vLLM model...")
    # 初始化vLLM模型
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        enforce_eager=False,
        max_model_len=41960,
        tensor_parallel_size=2,  # 根据您的GPU数量调整
        seed=0,
        dtype="bfloat16",
    )
    
    print(f"Starting image description generation...")
    print(f"Will process {selection_param*100 if isinstance(selection_param, float) else selection_param} images per folder")
    print(f"Each image will be described 5 times with varying parameters")
    
    process_folders(root_dir, folder_list, saved_folder=saved_folder, model=llm, 
                   processor=processor, selection_param=selection_param, 
                   seed=seed, dataset_split=dataset_split)
    
    print('Processing completed.')