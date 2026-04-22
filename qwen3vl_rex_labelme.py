import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
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

# 新增：Rex-Omni相关导入
from rex_omni import RexOmniWrapper

class AutoDataGroundingAgent:
    """自动数据标注智能体 - 整合到LabelMe脚本中"""
    
    def __init__(self, qwen_checkpoint_path="/mnt/data/lyf/Qwen3-VL-32B-Instruct", 
                 rex_model_path="/mnt/data/lyf/IDEA-Research/Rex-Omni"):
        
        # 初始化Qwen3-VL模型
        print("正在加载Qwen3-VL-32B模型...")
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_checkpoint_path)
        
        self.qwen_llm = LLM(
            model=qwen_checkpoint_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.6,
            enforce_eager=False,
            max_model_len=41960,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
            dtype="bfloat16",
        )
        
        self.qwen_sampling_params = SamplingParams(
            temperature=0,
            max_tokens=9280,
            top_k=-1,
            stop_token_ids=[],
        )
        print("Qwen3-VL-32B模型加载完成！")
        
        # 初始化Rex-Omni模型
        print("正在加载Rex-Omni模型...")
        self.rex_model = RexOmniWrapper(
            model_path=rex_model_path,
            backend="vllm",
            max_tokens=40960,
            gpu_memory_utilization=0.3,
            temperature=0.0
        )
        print("Rex-Omni模型加载完成！")
    
    def prepare_qwen_inputs(self, messages):
        """准备Qwen-VL输入"""
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.qwen_processor.image_processor.patch_size,
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
    
    def detect_categories_with_qwen(self, image):
        """使用Qwen3-VL检测图像中的视觉类别"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": "请列出这张图像中出现的所有视觉目标类别（如建筑，轿车，压路机，电动车，耕地等），用逗号分隔。只返回类别名称，不要有其他描述。"},
                    ],
                }
            ]
            
            inputs = [self.prepare_qwen_inputs(messages)]
            outputs = self.qwen_llm.generate(inputs, sampling_params=self.qwen_sampling_params)
            category_text = outputs[0].outputs[0].text.strip()
            
            # 解析类别文本
            categories = [cat.strip() for cat in category_text.split(',') if cat.strip()]
            return categories, category_text
            
        except Exception as e:
            return [], f"类别检测失败: {str(e)}"
    
    def detect_objects_with_rex_omni(self, image, categories):
        """使用Rex-Omni检测指定类别的对象"""
        try:
            # 执行Rex-Omni推理
            results = self.rex_model.inference(
                images=image,
                task="detection",
                categories=categories
            )
            
            result = results[0]
            if not result.get("success", False):
                raise RuntimeError(f"Rex-Omni推理失败: {result.get('error', '未知错误')}")
            
            predictions = result["extracted_predictions"]
            
            # 转换为标准格式的框数据
            boxes_data = []
            for category, detections in predictions.items():
                for detection in detections:
                    if detection["type"] == "box":
                        coords = detection["coords"]
                        boxes_data.append({
                            "xmin": coords[0],
                            "ymin": coords[1],
                            "xmax": coords[2],
                            "ymax": coords[3],
                            "label": category,
                            "confidence": detection.get("confidence", 0.0)
                        })
            
            return boxes_data, predictions
            
        except Exception as e:
            return [], f"对象检测失败: {str(e)}"
    
    def relabel_boxes_with_qwen(self, image, boxes_data, max_boxes_per_batch=10):
        """使用Qwen3-VL重新标定矩形框的类别和描述，支持分批处理"""
        try:
            if not boxes_data:
                return [], "没有检测到任何对象"
            
            width, height = image.size
            
            # 如果矩形框数量超过阈值，分批次处理
            if len(boxes_data) > max_boxes_per_batch:
                print(f"检测到 {len(boxes_data)} 个矩形框，超过阈值 {max_boxes_per_batch}，开始分批处理...")
                return self._relabel_boxes_batch(image, boxes_data, max_boxes_per_batch)
            
            # 单批次处理
            formatted_boxes = []
            for i, box in enumerate(boxes_data):
                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                
                # 转换为归一化坐标 [0, 1000]
                norm_x1 = int(xmin * 1000 / width)
                norm_y1 = int(ymin * 1000 / height)
                norm_x2 = int(xmax * 1000 / width)
                norm_y2 = int(ymax * 1000 / height)
                
                formatted_boxes.append({
                    "bbox_2d": [norm_x1, norm_y1, norm_x2, norm_y2],
                    "region_id": str(i + 1)  # 使用局部索引
                })
            
            # 构建重新标定提示词
            relabel_prompt = """请对以下每个区域进行详细描述，输出格式为：
            region_id: label|brief instance description

            要求：
            1. label使用中文名词
            2. description用中文简要描述该物体的特征、状态等
            3. 每个区域单独一行

            例如：
            1: 人/一个穿着红色衣服的在走路的年轻人
            2: 车辆/一辆停在路边的白色轿车

            请开始描述："""
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": relabel_prompt},
                        {
                            "type": "text", 
                            "text": f"以下是需要关注的区域坐标：{json.dumps(formatted_boxes, ensure_ascii=False)}"
                        }
                    ],
                }
            ]
            
            inputs = [self.prepare_qwen_inputs(messages)]
            outputs = self.qwen_llm.generate(inputs, sampling_params=self.qwen_sampling_params)
            relabel_result = outputs[0].outputs[0].text.strip()
            
            # 解析重新标定结果
            relabeled_boxes = self.parse_relabel_result(relabel_result, boxes_data)
            
            return relabeled_boxes, relabel_result
            
        except Exception as e:
            return [], f"重新标定失败: {str(e)}"
    
    def _relabel_boxes_batch(self, image, boxes_data, max_boxes_per_batch):
        """分批处理矩形框重新标定"""
        all_relabeled_boxes = []
        all_relabel_results = []
        width, height = image.size
        
        # 计算需要多少批次
        total_batches = (len(boxes_data) + max_boxes_per_batch - 1) // max_boxes_per_batch
        print(f"总共需要 {total_batches} 个批次进行处理")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * max_boxes_per_batch
            end_idx = min((batch_idx + 1) * max_boxes_per_batch, len(boxes_data))
            batch_boxes = boxes_data[start_idx:end_idx]
            
            print(f"正在处理第 {batch_idx + 1}/{total_batches} 批次，包含 {len(batch_boxes)} 个矩形框")
            
            # 格式化当前批次的框数据（使用局部索引）
            formatted_boxes = []
            for i, box in enumerate(batch_boxes):
                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                
                # 转换为归一化坐标 [0, 1000]
                norm_x1 = int(xmin * 1000 / width)
                norm_y1 = int(ymin * 1000 / height)
                norm_x2 = int(xmax * 1000 / width)
                norm_y2 = int(ymax * 1000 / height)
                
                formatted_boxes.append({
                    "bbox_2d": [norm_x1, norm_y1, norm_x2, norm_y2],
                    "region_id": str(i + 1)  # 使用局部索引（从1开始）
                })
            
            # 构建重新标定提示词
            relabel_prompt = """请对以下每个区域进行详细描述，输出格式为：
            region_id: label|brief instance description

            要求：
            1. label使用中文名词
            2. description用中文简要描述该物体的特征、状态等
            3. 每个区域单独一行

            例如：
            1: 人/一个穿着红色衣服的在走路的年轻人
            2: 车辆/一辆停在路边的白色轿车

            请开始描述："""
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": relabel_prompt},
                        {
                            "type": "text", 
                            "text": f"以下是需要关注的区域坐标：{json.dumps(formatted_boxes, ensure_ascii=False)}"
                        }
                    ],
                }
            ]
            
            inputs = [self.prepare_qwen_inputs(messages)]
            outputs = self.qwen_llm.generate(inputs, sampling_params=self.qwen_sampling_params)
            relabel_result = outputs[0].outputs[0].text.strip()
            
            # 解析当前批次的重新标定结果
            batch_relabeled_boxes = self.parse_relabel_result(relabel_result, batch_boxes)
            
            # 将当前批次的结果添加到总结果中
            all_relabeled_boxes.extend(batch_relabeled_boxes)
            
            # 在结果中添加批次信息
            batch_result_with_info = f"=== 批次 {batch_idx + 1}/{total_batches} (区域 {start_idx + 1}-{end_idx}) ===\n{relabel_result}"
            all_relabel_results.append(batch_result_with_info)
        
        # 合并所有批次的结果
        final_relabel_result = "\n\n".join(all_relabel_results)
        return all_relabeled_boxes, final_relabel_result
    
    def parse_relabel_result(self, relabel_text, original_boxes):
        """解析重新标定结果 - 按顺序匹配"""
        relabeled_boxes = []
        lines = [line for line in relabel_text.splitlines() if line.strip()]
        print(lines)
        print(original_boxes)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # 解析格式：region_id: label|description
            parts = line.split(':', 1)
            if len(parts) < 2:
                continue
            
            region_info = parts[1].strip()
            label = region_info
            # if '/' in region_info:
            #     label_part, description = region_info.split('/', 1)
            #     label = label_part.strip()
            #     description = description.strip()
            # else:
            #     label = region_info.strip()
            #     description = ""
            
            # 获取对应的原始框数据（按顺序匹配）
            if i < len(original_boxes):
                box_data = original_boxes[i].copy()
                box_data["relabel"] = label
                # box_data["description"] = description
                relabeled_boxes.append(box_data)
        
        return relabeled_boxes

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

# 修改后的保存为LabelMe格式的函数 - 适配新的标注格式
def save_labelme_format(relabeled_boxes, output_text, output_path, image_path, image_width, image_height):
    shapes = []
    for i, box in enumerate(relabeled_boxes):
        xmin, ymin, xmax, ymax = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        label = box.get("relabel", box.get("label", f"object_{i+1}"))
        # description = box.get("description", "")

        # 检查坐标合法性
        if xmin >= xmax or ymin >= ymax:
            print(f"Invalid coordinates for {label}: {[xmin, ymin, xmax, ymax]}. Skipping.")
            continue

        # 使用矩形形状类型
        points = [
            [xmin, ymin],  
            [xmax, ymax],
        ]
        
        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {
                # "description": description,
                "confidence": box.get("confidence", 0.0),
                "original_label": box.get("label", "")
            }
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
        # "description": output_text,
        "processing_method": "Rex-Omni + Qwen3-VL协作标注"
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=4, ensure_ascii=False)

# 修改后的处理每张图片函数 - 使用Rex-Omni和Qwen3-VL协作
def process_image(image_path, class_names, output_dir, agent, batch_threshold=10):
    # 检查该图片是否已经处理过
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return
    
    try:
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        
        print(f"开始处理图像: {os.path.basename(image_path)}")
        
        # 第一步：使用Qwen3-VL检测类别
        if class_names:
            categories = list(class_names.values()) if isinstance(class_names, dict) else class_names
            qwen_categories_output = f"使用手动输入类别: {', '.join(categories)}"
        else:
            categories, qwen_categories_output = agent.detect_categories_with_qwen(image)
            if not categories:
                print(f"未检测到任何类别: {image_path}")
                # 创建空标注文件
                save_labelme_format([], "未检测到任何类别", output_file, image_path, image_width, image_height)
                return
        
        print(f"检测到的类别: {categories}")
        
        # 第二步：使用Rex-Omni检测对象
        boxes_data, rex_raw_output = agent.detect_objects_with_rex_omni(image, categories)
        print(f"Rex-Omni检测到 {len(boxes_data)} 个对象")
        
        if not boxes_data:
            print(f"Rex-Omni未检测到任何对象: {image_path}")
            # 创建空标注文件
            save_labelme_format([], "Rex-Omni未检测到任何对象", output_file, image_path, image_width, image_height)
            return
        # 简单判断：如果矩形框超过120个，认为模型在胡说八道
        if len(boxes_data) > 120:
            print(f"警告：检测到异常多的目标({len(boxes_data)})，跳过处理")
            # 创建空标注文件
            save_labelme_format([], f"检测目标过多({len(boxes_data)}个)，疑似模型异常", output_file, image_path, image_width, image_height)
            return        
        # 第三步：使用Qwen3-VL重新标定
        relabeled_boxes, qwen_relabel_text = agent.relabel_boxes_with_qwen(
            image, boxes_data, max_boxes_per_batch=batch_threshold
        )
        print(f"Qwen3-VL重新标定了 {len(relabeled_boxes)} 个对象")
        
        # 将 LabelMe 格式保存到文件
        save_labelme_format(relabeled_boxes, qwen_relabel_text, output_file, image_path, image_width, image_height)
        print(f"处理完成并保存: {image_path}")
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出现错误: {str(e)}")
        # 创建错误标注文件
        try:
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            save_labelme_format([], f"处理错误: {str(e)}", output_file, image_path, image_width, image_height)
        except:
            pass

# 修改后的处理文件夹函数，支持选择处理train, val, test
def process_folder(root_dir, folder, saved_folder='pred', agent=None, selection_param=1.0, seed=None, dataset_split="all", batch_threshold=10):
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
            
            image_files = [os.path.join(images_folder, os.path.basename(image)) for line in lines for image in json.loads(line.strip())["images"]]
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
            process_image(image_path, class_names, pred_folder, agent, batch_threshold)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    print(f"All predictions saved in: {pred_folder}")

# 修改后的处理文件夹列表函数，支持灵活设置百分比或影像个数
def process_folders(root_dir, folder_list, saved_folder='pred', agent=None, selection_param=1.0, seed=None, dataset_split="all", batch_threshold=10):
    for i, folder in enumerate(folder_list):
        print(f"Processing folder: {folder}")

        try:
            process_folder(root_dir, folder, saved_folder=saved_folder, agent=agent,
                           selection_param=selection_param, seed=seed, dataset_split=dataset_split, batch_threshold=batch_threshold)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

# 主程序入口
if __name__ == "__main__":
    root_dir = "/mnt/data/lyf/datasets"
    folder_list = ["1431_part1"]
    saved_folder = 'labelme'

    dataset_split = "all"  # 可以设置为 "train", "val", "test", 或 "all"
    selection_param = 1.0
    seed = 52 #42 
    batch_threshold = 10  # 分批处理阈值

    # 初始化自动数据标注智能体
    print("正在初始化自动数据标注智能体...")
    agent = AutoDataGroundingAgent()
    print("智能体初始化完成！")

    process_folders(root_dir, folder_list, saved_folder=saved_folder, agent=agent, 
                   selection_param=selection_param, seed=seed, dataset_split=dataset_split, batch_threshold=batch_threshold)
    print('Processing completed.')