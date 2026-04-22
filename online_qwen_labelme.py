import os
import json
import random
import numpy as np
import base64
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import re
import shutil


def base64_from_cv2_image(image, prefix=False):
    assert isinstance(image, np.ndarray), "image should be np.ndarray"
    image = cv2.imencode('.jpg', image)[1]
    img_b64 = str(base64.b64encode(bytes(image)))[2:-1]
    if prefix:
        img_b64 = 'data:image/jpeg;base64,' + img_b64
    return img_b64


def  openai_api_request_det(text: str, img:None) -> str:
    # Set up API endpoint and headers
    #  qwen25-vl 图像，解析
    
    SERVICE_ID = "modelservice-bf9e"
    PROJECT_ID = "1"
    default_headers = {
        "X-TC-Action": "/v1/chat/completions",
        "X-TC-Version": "2020-10-01",
        "X-TC-Service": SERVICE_ID,
        "X-TC-Project": PROJECT_ID,
        "Content-Type": "application/json",
        }
    
    
    api_key = "EMPTY"
    base_url = "http://10.19.52.246/gateway/v1"
    
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers,
    )
    img_b64 = base64_from_cv2_image(img, prefix=False)
    messages = [  
            {  
                "role": "user",  
                "content": [  
                    {  
                        "type": "text",  
                        "text": str(text),  
                    },  
                    {  
                        "type": "image_url",  
                        "image_url": {  
                            "url": f"data:image/jpeg;base64,{img_b64}"  
                        }  
                    }  
                ]  
            }  
        ]
    completion = client.chat.completions.create(
        model="qwen25-vl",
        messages=messages
    )
    content = completion.choices[0].message.content 
    return content
# 修改后的保存为LabelMe格式的函数
def save_labelme_format(predictions, description, output_path, image_path, image_width, image_height):
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
        "description":description
    }

    with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=4, ensure_ascii=False)

def openai_api_request_parse(content):
    
    SERVICE_ID = "modelservice-e670"
    PROJECT_ID = "1"
    default_headers = {
        "X-TC-Action": "/v1/chat/completions",
        "X-TC-Version": "2020-10-01",
        "X-TC-Service": SERVICE_ID,
        "X-TC-Project": PROJECT_ID,
        "Content-Type": "application/json",
        }
    
    
    api_key = "EMPTY"
    base_url = "http://10.37.1.27/gateway/v1"
    
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers,
    )
    txt = '下列文本中是从视觉大模型中提取得到的检测结果，请从下列文本中解析出出每个矩形框以及对应的类别，并将上述结果标签和box相近的合并后输出，输出格式应为:{{\"bbox_2d\": [x1,y1,x2,y2],\"label\":\"aaa\",\"sub_lable\":\"bbb\"}}，其中aaa为目标名称, bbb为更具体的子类名,不过bbb可能为空，如果为空的话，请保持为空，你不要给用其他文字代替，\
    如果文本中没有矩形框，就返回空，另外需要注意的是有可能有些矩形框的类别有些问题，对于这样的矩形框，请你校正一下类别名称。以下是文本：'+content
    messages = [  
            {  
                "role": "user",  
                "content": [  
                    {  
                        "type": "text",  
                        "text": txt
                    },  
                ]  
            }  
        ]
    completion = client.chat.completions.create(
        model="qwen72b",
        messages=messages
    )
    content = completion.choices[0].message.content 
    return content

def deepseek_request_parse(content):
    txt = '下列文本中是从视觉大模型中提取得到的检测结果，请从下列文本中解析出出每个矩形框以及对应的类别，并将上述结果标签和box相近的合并后输出，输出格式应为:{{\"bbox_2d\": [x1,y1,x2,y2],\"label\":\"aaa\",\"sub_lable\":\"bbb\"}}，其中aaa为目标名称, bbb为更具体的子类名,不过bbb可能为空，如果为空的话，请保持为空，你不要给用其他文字代替，\
    如果文本中没有矩形框，就返回空，另外需要注意的是有可能有些矩形框的类别有些问题，对于这样的矩形框，请你校正一下类别名称。以下是文本：'+content
    
    url = 'http://121.37.99.52:41025/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": txt
            }
        ],
        "model": "DeepSeek-R1",
        "temperature": 0,
        "max_tokens": 4096
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if result and 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        
    except requests.RequestException as e:
        print(f"请求出错: {e}")
        return None
    except ValueError as e:
        print(f"解析响应为 JSON 时出错: {e}")
        return None

# 解析响应中的检测框
def parse_response_boxes(response, box_offsets=[0, 0]):
    """
    从响应文本中解析检测框坐标并按比例还原，并且支持多个类别。
    """
    # 创建一个字典，键是类别，值是框
  
    # 正则表达式修改为支持多类别的解析，输出格式：{ "bbox_2d": [x1, y1, x2, y2], "label": "{category_name}" }
    potential_lines = []
    response = response.replace('\n', ',')
    for line in response:
        # print("line: ", line)
        if '{"bbox_2d":' in line:
            potential_lines.append(line)
    print("potential_lines: ", line,  potential_lines)

    # 调整正则表达式，使其更具包容性
    '''box_pattern = r'\{"bbox_2d":\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"([^"]+)"\}'

    matches = []
    for line in potential_lines:
        match = re.search(box_pattern, line)
        if match:
            matches.append(match.groups())'''
    box_pattern  = r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"(?:,\s*"sub lable":\s*"(.*?)")?'
    #r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"'


    matches = re.findall(box_pattern, response)
    
    # 解析每一个框
    category_boxes = {}
    for match in matches:
        x1, y1, x2, y2, class_name, sub_classname = match
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = x1 + box_offsets[0]
        y1 = y1 + box_offsets[1]
        x2 = x2 + box_offsets[0]
        y2 = y2 + box_offsets[1]
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

# 处理单次推理
def process_single_detection(img, text):
    content = openai_api_request_det(text, img)
    parsed_content = deepseek_request_parse(content) #也可以用qwen解析，替换为openai_api_request_parse(content)
    return parse_response_boxes(parsed_content),content

def nms_by_class(boxes, iou_threshold=0.5):
    """
    执行按类别的非极大值抑制（NMS）使用 OpenCV NMS
    boxes 格式: [(label, [x1, y1, x2, y2]), ...]
    返回：经过NMS后的结果
    """
    # 按类别分组
    class_boxes = {}
    for label, box in boxes:
        if label not in class_boxes:
            class_boxes[label] = []
        class_boxes[label].append((box))

    final_boxes = []

    # 对每个类别分别执行NMS
    for label, boxes_in_class in class_boxes.items():
        boxes_in_class_np = np.array([box for box in boxes_in_class], dtype=np.float32)
        scores = np.ones(len(boxes_in_class_np))  # 简单地设定所有框的得分为 1

        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(boxes_in_class_np.tolist(), scores.tolist(), score_threshold=0, nms_threshold=iou_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append((label, boxes_in_class[i]))

    return final_boxes

# 处理每张图片
def process_image(image_path, class_names, output_dir, model_num = 5):
    # 检查该图片是否已经处理过
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return
    try:
    # 加载图像
        Img = cv2.imread(image_path)
        image_height, image_width = Img.shape[:2]
        
        # 构建 query，一次性询问所有类别
        if class_names:
            class_list_str = "，".join(class_names.values())
            category_name = f"请检测图像中包含的{class_list_str}等目标，并以坐标形式返回它们的位置，如果不存在则不用输出坐标，输出格式应为:{{\"bbox_2d\": [x1,y1,x2,y2],\"label\":\"XXX\"}}，其中XXX为目标名称。"
        else:
             category_name = "第一步，创建详细的图像描述，尽可能详细地描述给定图像的内容。描述应包括物体类型、纹理和颜色、物体的组成部分、物体的动作、精确的物体位置、文本内容，并仔细核对物体之间的相对位置等信息。只描述能从图像中确定的内容，不要描述想象的内容。不要以列表形式逐项描述内容，尽可能减少美学方面的描述。第二步，基于第一步的描述提取出这张图中出现的实体目标并输出；第三步，输出第二步提取的类别并输出对应的目标框，以坐标形式返回它们的位置，如果不存在则不用输出坐标，输出格式应为:{{\"bbox_2d\": [x1,y1,x2,y2],\"label\":\"aaa\",\"sub_lable\":\"bbb\"}}，其中aaa为目标名称, bbb为更具体的子类名。"   
        
    
        
        results = []
        for _ in range(model_num):
            result = process_single_detection(Img, category_name)
            results.append(result)
        # 解析检测结果
        parsed_boxes = [box for result in results for box in result[0]]
        final_boxes = nms_by_class(parsed_boxes)
        if not final_boxes:
            # 如果没有检测框，生成一个空文件
            open(output_file, 'w').close()
            print(f"No detections for {image_path}. Created empty file: {output_file}")
            return
        #shutil.copy2(image_path, output_dir)
        # 将 LabelMe 格式保存到文件
        descriptions = [desc for _, desc in results]
        save_labelme_format(final_boxes, description, output_file, image_path, image_width, image_height)
        print(f"Processed and saved: {image_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")   

# 修改后的处理文件夹函数，支持选择处理train, val, test
def process_folder(root_dir, folder, saved_folder='pred', model_num=5, selection_param=1.0, seed=None, dataset_split="all"):
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
            process_image(image_path, class_names, pred_folder, model_num)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    print(f"All predictions saved in: {pred_folder}")

# 修改后的处理文件夹列表函数，支持灵活设置百分比或影像个数
def process_folders(root_dir, folder_list, saved_folder='pred', selection_param=1.0, seed=None, dataset_split="all"):
    for i, folder in enumerate(folder_list):
        print(f"Processing folder: {folder}")

        try:
            process_folder(root_dir, folder, saved_folder=saved_folder,
                           selection_param=selection_param, seed=seed, dataset_split=dataset_split)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

# 主程序入口
if __name__ == "__main__":
    
    root_dir = "/data1/lyf/data/mulobj20250201_20250212" #root dir，下面会有一些子文件件如615,616
    folder_list = ["615","616"] #要用于标注的子文件夹名称
    saved_folder = 'labelme' #无需修改，会将标注结果存放于folder_list下的labelme文件夹下
    num_det = 5
    dataset_split = "all"  # 可以设置为 "train", "val", "test", 或 "all"， 要对folder_list下的哪一个子文件"train", "val", "test"标注，也可以是全部都标注即为'all'
    selection_param = 1.0 #要标注多少比例的图像，可默认是全部标注，即1.0
    seed = 42 
    process_folders(root_dir, folder_list, saved_folder=saved_folder, model_num=num_det, selection_param=selection_param, seed=seed, dataset_split=dataset_split)
    print('Processing completed.')
