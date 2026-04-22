import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
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

    with open(output_path, 'w') as f:
        json.dump(annotation, f, indent=4)

# 解析 response 中的 bbox 和类别
def parse_response_boxes(response):
    """
    从响应文本中解析检测框坐标并按比例还原，并且支持多个类别。
    """
    box_pattern = r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"(?:,\s*"sub lable":\s*"(.*?)")?'
    matches = re.findall(box_pattern, response[0])

    category_boxes = {}
    for match in matches:
        x1, y1, x2, y2, class_name, sub_classname = match
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
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
def process_image(image_path, class_names, output_dir, model):
    # 检查该图片是否已经处理过
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return
    
    # 加载图像
    data = Image.open(image_path).convert("RGB")
    image_width, image_height = data.size
    
    # 构建 query，一次性询问所有类别
    if class_names:
        class_list_str = "，".join(class_names.values())
        category_name = f"请检测图像中包含的{class_list_str}等目标，并以坐标形式返回它们的位置，如果不存在则不用输出坐标，输出格式应为:{{\"bbox_2d\": [x1,y1,x2,y2],\"label\":\"XXX\"}}，其中XXX为目标名称。"
    else:
        category_name = f"你是一个图像解译助手，请先详细描述这张图，然后检测所有目标（你描述中中出现的目标和其他可见目标），并以坐标形式返回它们的位置，如果不存在则不用输出坐标，输出格式应为:{{\"bbox_2d\": [x1,y1,x2,y2],\"label\":\"aaa\",\"sub lable\":\"bbb\"}}，其中aaa为目标名称, bbb为更具体的子类名"   
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    placeholder = "<|image_pad|>"
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{category_name}<|im_end|>\n"
              "<|im_start|>assistant\n")
    
    # 构建推理请求
    inputs = {
            "prompt": prompt,
            "multi_modal_data": {'image': data}
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
    #shutil.copy2(image_path, output_dir)
    # 将 LabelMe 格式保存到文件
    save_labelme_format(parsed_boxes, output_text, output_file, image_path, image_width, image_height)
    print(f"Processed and saved: {image_path}")

# 修改后的处理文件夹函数，支持选择处理train, val, test
def process_folder(root_dir, folder, saved_folder='pred', model=None, selection_param=1.0, seed=None, dataset_split="all"):
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
            process_image(image_path, class_names, pred_folder, model)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

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
    root_dir = "/mnt/data/lzw/data_label_qw/mulobj_part7_20250327_20250331"
    folder_list = ["696","697"]
	#["678","679","680","681"] #678  679  680  681 和 "682","683","684", "685" 被暂停，先跑梁博的
    saved_folder = 'labelme'

    dataset_split = "all"  # 可以设置为 "train", "val", "test", 或 "all"
    selection_param = 1.0
    seed = 52 #42 

    model_name = "/mnt/data/lyf/qwenvl-2.5-72b"
    llm = LLM(model=model_name, tensor_parallel_size=4)

    process_folders(root_dir, folder_list, saved_folder=saved_folder, model=llm, selection_param=selection_param, seed=seed, dataset_split=dataset_split)
    print('Processing completed.')
