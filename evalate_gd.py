import os
import json
import re
import yaml
from tqdm import tqdm
from PIL import Image
import torch
import fiftyone as fo
# 解析 response 中的 bbox
def parse_response_boxes(response, scale_x, scale_y):
    box_pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    boxes = []
    matches = re.findall(box_pattern, response)
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        boxes.append([x1, y1, x2, y2])
    return boxes


def load_yolo_file(class_list_path):
    with open(class_list_path, 'r', encoding='utf-8') as f:
        class_list = [line.strip() for line in f.readlines() if line.strip()]
    return class_list

def load_yolo_labels(label_path, classes, pred=False):
    detections = []
    lines = load_yolo_file(label_path)
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        label = classes[class_id]
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])
        bounding_box = [
            x_center - bbox_width / 2,
            y_center - bbox_height / 2,
            bbox_width,
            bbox_height,
        ]
        if pred==True:
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=bounding_box,
                    confidence=1
                )
            )
        else:
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=bounding_box,
                )
            )
            
    return detections


def evaluate_predictions(dataset_name, folder_list, root_dir, iou_threshold=0.5, saved_folder='pred'):
    dataset = fo.Dataset(name=dataset_name, overwrite=True)
    samples = []
    class_names = None

    for folder in folder_list:
        print(f"Processing folder: {folder}")
        pred_dir = os.path.join(root_dir, folder, saved_folder)
        gt_dir = os.path.join(root_dir, folder, "labels")
        images_dir = os.path.join(root_dir, folder, "images")
        input_folder = os.path.join(root_dir, folder)
        if not os.path.exists(pred_dir) or not os.path.exists(gt_dir) or not os.path.exists(images_dir):
            print(f"Warning: Missing required folders in {folder}. Skipping...")
            continue

        
        yaml_files = ['data.yaml', 'dataset.yaml']
        yaml_path = next((os.path.join(input_folder, f) for f in yaml_files if os.path.exists(os.path.join(input_folder, f))), None)
    
        
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                class_names = data_config.get("names", [])
        else:
            print(f"Warning: dataset.yaml not found in {folder}. Using default class names.")
            continue

        for file_name in os.listdir(gt_dir):
            if file_name.endswith(".txt"):
                image_name = file_name.replace(".txt", ".jpg")
                image_path = os.path.join(images_dir, image_name)
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found for {file_name}")
                    continue
                gt_label_path = os.path.join(gt_dir, file_name)
                pred_label_path = os.path.join(pred_dir, file_name)
                if not os.path.exists(pred_label_path):  # 如果没有预测文件，跳过此图像
                    print(f"Warning: No prediction file for {image_name}. Skipping...")
                    continue
                gt_detections = load_yolo_labels(gt_label_path, class_names)
                pred_detections = load_yolo_labels(pred_label_path, class_names, pred=True)

                sample = fo.Sample(filepath=image_path)
                sample["ground_truth"] = fo.Detections(detections=gt_detections)
                sample["predictions"] = fo.Detections(detections=pred_detections)
                samples.append(sample)

    dataset.add_samples(samples)

    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="evaluation",
        iou=iou_threshold,
        compute_mAP=True,
    )

    print("\nEvaluation Report:")
    results.print_report()

    session = fo.launch_app(dataset,address="0.0.0.0", port=8002)
    session.wait()

    return results


root_dir = "/data1/lyf/formate_t/coco_format"
folder_list = ["zixingche"]
saved_folder = 'pred_multi_engine_only_qwen25/val'

results = evaluate_predictions(
    dataset_name="my_combined",
    folder_list=folder_list,
    root_dir=root_dir,
    iou_threshold=0.5,
    saved_folder=saved_folder
)

print('done')
