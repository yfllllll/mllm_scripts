import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
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

def extract_json_from_response(response_text):
    """从模型响应中提取JSON格式数据"""
    try:
        # 尝试直接解析整个响应
        return json.loads(response_text)
    except json.JSONDecodeError:
        # 如果失败，尝试提取JSON部分
        try:
            # 查找JSON开始和结束的位置
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # 如果找不到大括号，返回空结构
                return None
        except:
            return None

# 修改后的保存为LabelMe格式的函数（适配VQA数据）
def save_labelme_format(vqa_data, output_path, image_path, image_width, image_height):
    """保存VQA数据到LabelMe格式"""
    
    annotation = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],  # 空列表，因为我们不标注目标
        "imagePath": os.path.basename(image_path),  # 图像文件名
        "imageData": None,  # Base64图像数据（可选）
        "imageHeight": image_height,
        "imageWidth": image_width,
        "vqa_data": vqa_data,  # 存储VQA数据
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 添加时间戳
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=4, ensure_ascii=False)

# 处理每张图片，生成VQA数据
def process_image(image_path, output_dir, model, processor):
    # 检查该图片是否已经处理过
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    
    # VQA提示词
    vqa_prompt = """请你基于图像内容自问自答5个问题，用于构建VQA数据集。要求：
1. 问题需基于图像中的具体目标或场景
2. 包含以下类型的问题（每种至少一个）：
   - 属性类（如颜色、大小、数量）
   - 动作类（如人物/动物在做什么）
   - 关系类（如物体之间的位置关系）
   - 场景理解类（如发生了什么事）
   - 开放推理类（如图像可能的地点、时间、原因等）
3. 答案应简洁准确，基于图像内容

请返回严格的JSON格式，结构如下：
{
  "image_description": "简要描述图像内容（1-2句话）",
  "qa_pairs": [
    {
      "question": "问题1文本",
      "answer": "答案1文本",
      "question_type": "属性/动作/关系/场景/推理"
    },
    ...（共5组）
  ]
}

只返回JSON对象，不要额外解释。"""
    
    print(f"Processing {image_path}...")
    
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
                    {"type": "text", "text": vqa_prompt},
                ],
            }
        ]
        
        # 准备输入
        inputs = prepare_inputs_for_vllm(messages, processor)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,  # 稍微提高多样性
            max_tokens=2048,  # VQA需要更多token
            top_p=0.9,
            stop_token_ids=[],
        )
        
        # 模型推理
        outputs = model.generate([inputs], sampling_params)
        output_text = outputs[0].outputs[0].text.strip()
        
        # 提取并解析JSON
        parsed_data = extract_json_from_response(output_text)
        
        if parsed_data:
            # 验证数据结构
            if "image_description" in parsed_data and "qa_pairs" in parsed_data:
                # 确保有5个QA对
                if len(parsed_data["qa_pairs"]) != 5:
                    print(f"  Warning: Expected 5 QA pairs, got {len(parsed_data['qa_pairs'])}")
                
                # 添加元数据
                vqa_data = {
                    "image_description": parsed_data["image_description"],
                    "qa_pairs": parsed_data["qa_pairs"],
                    "model_info": {
                        "model_name": "Qwen3-VL-32B-Instruct",
                        "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "temperature": sampling_params.temperature
                    }
                }
                
                print(f"  Successfully generated VQA data")
                print(f"  Image description: {parsed_data['image_description'][:100]}...")
                for i, qa in enumerate(parsed_data["qa_pairs"][:2]):  # 只显示前两个
                    print(f"  Q{i+1}: {qa['question'][:50]}...")
                
                # 将VQA数据保存到文件
                save_labelme_format(vqa_data, output_file, image_path, image_width, image_height)
                print(f"  Saved VQA data to: {output_file}")
            else:
                print(f"  Error: Missing required fields in response")
                # 创建错误占位数据
                create_error_placeholder(output_file, image_path, image_width, image_height, output_text)
        else:
            print(f"  Error: Failed to parse JSON from response")
            print(f"  Raw output: {output_text[:200]}...")
            # 创建错误占位数据
            create_error_placeholder(output_file, image_path, image_width, image_height, output_text)
            
    except Exception as e:
        print(f"  Error processing image: {e}")
        import traceback
        traceback.print_exc()
        # 创建错误占位数据
        create_error_placeholder(output_file, image_path, image_width, image_height, str(e))

def create_error_placeholder(output_file, image_path, image_width, image_height, error_msg):
    """创建错误占位数据"""
    # error_data = {
    #     "image_description": f"Error generating VQA data: {error_msg[:100]}",
    #     "qa_pairs": [],
    #     "error": True,
    #     "error_message": str(error_msg)[:500],
    #     "model_info": {
    #         "model_name": "Qwen3-VL-32B-Instruct",
    #         "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    #         "note": "Failed to generate proper VQA data"
    #     }
    # }
    # save_labelme_format(error_data, output_file, image_path, image_width, image_height)
    # print(f"  Saved error placeholder to: {output_file}")
    print(f"  Error placeholder: {error_msg[:100]}...")
    pass

# 处理文件夹函数（保持不变，只修改函数名）
def process_folder(root_dir, folder, saved_folder='vqa_data', model=None, processor=None, 
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
    
    print(f"All VQA data saved in: {pred_folder}")

# 处理文件夹列表函数
def process_folders(root_dir, folder_list, saved_folder='vqa_data', model=None, processor=None, 
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
    saved_folder = 'vqa_data'  # 修改输出文件夹名称
    
    dataset_split = "all"  # 可以设置为 "train", "val", "test", 或 "all"
    selection_param = 1.0  # 处理100%的图片，可以根据需要调整
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
    
    print(f"Starting VQA data generation...")
    print(f"Will process {selection_param*100 if isinstance(selection_param, float) else selection_param} images per folder")
    print(f"Each image will generate 5 QA pairs")
    
    process_folders(root_dir, folder_list, saved_folder=saved_folder, model=llm, 
                   processor=processor, selection_param=selection_param, 
                   seed=seed, dataset_split=dataset_split)
    
    print('VQA data generation completed.')