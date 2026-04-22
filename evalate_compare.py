import os
import json
import yaml
from tqdm import tqdm
import fiftyone as fo

# ==================== 支持的图像扩展名 ====================
SUPPORTED_IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def find_image_file(directory, stem):
    """在目录中查找与 stem 匹配的图像文件，返回完整文件名（含扩展名），若找不到返回 None"""
    for ext in SUPPORTED_IMAGE_EXTS:
        fname = stem + ext
        if os.path.exists(os.path.join(directory, fname)):
            return fname
    return None

def load_labelme_gt(json_path, class_names):
    """从 LabelMe JSON 加载真实标注，返回 fo.Detection 列表"""
    if os.path.getsize(json_path) == 0:
        return []  # 空文件，无检测框
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    w_img = data['imageWidth']
    h_img = data['imageHeight']
    detections = []
    for shape in data.get('shapes', []):
        label = shape['label']
        base_label = label.split('/')[0]
        if base_label not in class_names:
            print(f"Warning: label '{label}' not in class_names, skipping")
            continue
        points = shape['points']
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            width = x2 - x1
            height = y2 - y1
            xc = (x1 + x2) / 2.0 / w_img
            yc = (y1 + y2) / 2.0 / h_img
            w = width / w_img
            h = height / h_img
            bbox = [xc - w/2, yc - h/2, w, h]
            detections.append(fo.Detection(label=base_label, bounding_box=bbox))
        else:
            print(f"Warning: shape with unsupported points format: {points}")
    return detections

def load_yolo_labels(label_path, class_names):
    """从 YOLO 格式 .txt 加载真实标注，返回 fo.Detection 列表"""
    detections = []
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        if class_id >= len(class_names):
            print(f"Warning: class_id {class_id} out of range")
            continue
        label = class_names[class_id]
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        bbox = [xc - w/2, yc - h/2, w, h]
        detections.append(fo.Detection(label=label, bounding_box=bbox))
    return detections

def load_labelme_predictions(json_path, class_names):
    """从 LabelMe JSON 加载预测结果，返回带置信度（默认为1.0）的 fo.Detection 列表"""
    dets = load_labelme_gt(json_path, class_names)
    for d in dets:
        d.confidence = 1.0
    return dets

def load_yolo_predictions(label_path, class_names):
    """从 YOLO 格式 .txt 加载预测结果，返回带置信度（默认为1.0）的 fo.Detection 列表"""
    dets = load_yolo_labels(label_path, class_names)
    for d in dets:
        d.confidence = 1.0
    return dets

def load_predictions(pred_dir, class_names, pred_format='yolo'):
    """
    加载预测目录中的所有预测结果，返回字典 {image_filename: fo.Detections}
    图像文件名使用实际存在的扩展名。
    """
    pred_dict = {}
    if pred_format == 'yolo':
        for file in os.listdir(pred_dir):
            if not file.endswith('.txt'):
                continue
            stem = os.path.splitext(file)[0]
            img_file = find_image_file(pred_dir, stem)
            if img_file is None:
                print(f"Warning: No image found for prediction file {file}, skipping.")
                continue
            pred_path = os.path.join(pred_dir, file)
            dets = load_yolo_predictions(pred_path, class_names)
            pred_dict[img_file] = dets
    elif pred_format == 'labelme':
        for file in os.listdir(pred_dir):
            if not file.endswith('.json'):
                continue
            stem = os.path.splitext(file)[0]
            img_file = find_image_file(pred_dir, stem)
            if img_file is None:
                print(f"Warning: No image found for prediction file {file}, skipping.")
                continue
            json_path = os.path.join(pred_dir, file)
            dets = load_labelme_predictions(json_path, class_names)
            pred_dict[img_file] = dets
    else:
        raise ValueError(f"Unsupported pred_format: {pred_format}")
    return pred_dict

def build_samples_for_service(
    root_dir, folder_list, gt_format, class_names,
    pred_dir=None, only_with_preds=True
):
    """
    根据预测目录构建样本列表（每个样本包含 ground_truth 字段）。
    图像路径必须来自预测目录下的图像副本，若不存在则跳过该图像。
    """
    samples = []
    for folder in folder_list:
        # GT 目录
        if gt_format == 'yolo':
            gt_dir = os.path.join(root_dir, folder, 'labels')
        elif gt_format == 'labelme':
            gt_dir = os.path.join(root_dir, folder, 'gt_labelme')
        else:
            raise ValueError(f"Unsupported gt_format: {gt_format}")

        # 预测目录（必须提供）
        if not pred_dir:
            raise ValueError("pred_dir must be provided when building samples from prediction directory.")
        if os.path.isabs(pred_dir):
            pred_dir_abs = pred_dir
        else:
            pred_dir_abs = os.path.join(root_dir, folder, pred_dir)

        if not os.path.exists(pred_dir_abs):
            print(f"Warning: Prediction directory {pred_dir_abs} not found for folder {folder}, skipping.")
            continue

        # 获取所有预测文件的基础名集合（如果 only_with_preds）
        pred_stems = set()
        if only_with_preds:
            for f in os.listdir(pred_dir_abs):
                if f.endswith(('.json', '.txt')):
                    pred_stems.add(os.path.splitext(f)[0])

        # 遍历 GT 文件
        if gt_format == 'yolo':
            gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
        else:
            gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]

        for gt_file in gt_files:
            img_stem = os.path.splitext(gt_file)[0]
            if only_with_preds and img_stem not in pred_stems:
                continue

            # 查找实际图像文件
            img_file = find_image_file(pred_dir_abs, img_stem)
            if img_file is None:
                print(f"Warning: No image found for {img_stem} in {pred_dir_abs}, skipping.")
                continue

            img_path = os.path.join(pred_dir_abs, img_file)

            # 加载 GT
            gt_path = os.path.join(gt_dir, gt_file)
            if gt_format == 'yolo':
                dets = load_yolo_labels(gt_path, class_names)
            else:
                dets = load_labelme_gt(gt_path, class_names)

            sample = fo.Sample(filepath=img_path)
            sample['ground_truth'] = fo.Detections(detections=dets)
            samples.append(sample)

    return samples

def evaluate_single_service(
    service_name, dataset_name,
    root_dir, folder_list, gt_format, class_names,
    pred_dir, pred_format, output_dir,
    only_with_preds=True
):
    # 构建样本（图像必须位于预测目录下）
    samples = build_samples_for_service(
        root_dir, folder_list, gt_format, class_names,
        pred_dir=pred_dir, only_with_preds=only_with_preds
    )
    if not samples:
        print(f"No samples built for {service_name} (possibly no predictions?).")
        return None, None

    # 加载预测字典
    if os.path.isabs(pred_dir):
        pred_dir_abs = pred_dir
    else:
        if folder_list:
            pred_dir_abs = os.path.join(root_dir, folder_list[0], pred_dir)
        else:
            pred_dir_abs = pred_dir

    if not os.path.exists(pred_dir_abs):
        print(f"Warning: Prediction directory {pred_dir_abs} not found for {service_name}")
        return None, None

    pred_dict = load_predictions(pred_dir_abs, class_names, pred_format)

    for sample in samples:
        img_name = os.path.basename(sample.filepath)
        pred_dets = pred_dict.get(img_name, [])
        sample['predictions'] = fo.Detections(detections=pred_dets)

    dataset = fo.Dataset(name=f"{dataset_name}_{service_name}", overwrite=True)
    dataset.add_samples(samples)
    eval_key = f"eval_{service_name}"
    results = dataset.evaluate_detections(
        'predictions',
        gt_field='ground_truth',
        method='coco',
        eval_key=eval_key,
        compute_mAP=True,
        iou_threshs=[0.5]
    )

    if output_dir:
        out_subdir = os.path.join(output_dir, service_name)
        os.makedirs(out_subdir, exist_ok=True)
        report_file = os.path.join(out_subdir, "report.txt")
        with open(report_file, 'w') as f:
            f.write(f"Evaluation Report for {service_name}\n")
            f.write("COCO standard evaluation (IoU thresholds 0.5)\n\n")
            results.print_report(classes=class_names)
            f.write("\n=== Summary Metrics ===\n")
            f.write(f"mAP (COCO): {results.mAP()}\n")
            f.write(f"mAR (COCO): {results.mAR()}\n")
            tp = dataset.sum(f"{eval_key}_tp")
            fp = dataset.sum(f"{eval_key}_fp")
            fn = dataset.sum(f"{eval_key}_fn")
            f.write(f"Total TP: {tp}\n")
            f.write(f"Total FP: {fp}\n")
            f.write(f"Total FN: {fn}\n")

        metrics = results.metrics()
        json_file = os.path.join(out_subdir, "metrics.json")
        with open(json_file, 'w') as f:
            json.dump({
                'mAP': results.mAP(),
                'mAR': results.mAR(),
                'metrics': metrics,
                'classes': class_names,
                'total_tp': tp,
                'total_fp': fp,
                'total_fn': fn,
            }, f, indent=2)

        print(f"Saved evaluation results for {service_name} to {out_subdir}")

    return results, samples

def evaluate_services(
    dataset_name, folder_list, root_dir,
    pred1_dir=None, pred1_format='yolo',
    pred2_dir=None, pred2_format='yolo',
    gt_format='yolo', output_dir=None,
    only_with_preds=True
):
    # 加载类别名称
    class_names = None
    for folder in folder_list:
        input_folder = os.path.join(root_dir, folder)
        yaml_files = ['data.yaml', 'dataset.yaml']
        yaml_path = next((os.path.join(input_folder, f) for f in yaml_files if os.path.exists(os.path.join(input_folder, f))), None)
        if yaml_path:
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                names_dict = data_config.get('names', {})
                class_names = [names_dict[i] for i in sorted(names_dict.keys())]
            break
    if not class_names:
        raise ValueError("Could not find class names in any folder. Please provide class_names explicitly.")

    results1, samples1 = None, None
    results2, samples2 = None, None

    if pred1_dir is not None:
        results1, samples1 = evaluate_single_service(
            'service1', dataset_name,
            root_dir, folder_list, gt_format, class_names,
            pred1_dir, pred1_format, output_dir,
            only_with_preds
        )
    if pred2_dir is not None:
        results2, samples2 = evaluate_single_service(
            'service2', dataset_name,
            root_dir, folder_list, gt_format, class_names,
            pred2_dir, pred2_format, output_dir,
            only_with_preds
        )

    if results1 and results2 and output_dir:
        comp_file = os.path.join(output_dir, "comparison.txt")
        with open(comp_file, 'w') as f:
            f.write("=== Comparison between Service1 and Service2 ===\n\n")
            f.write("COCO standard evaluation (IoU thresholds 0.5)\n\n")
            f.write(f"{'Metric':<15} {'Service1':<20} {'Service2':<20} {'Difference':<15}\n")
            m1 = results1.mAP()
            m2 = results2.mAP()
            diff = m2 - m1
            f.write(f"{'mAP (COCO)':<15} {m1:<20.4f} {m2:<20.4f} {diff:<15.4f}\n")
            m1 = results1.mAR()
            m2 = results2.mAR()
            diff = m2 - m1
            f.write(f"{'mAR (COCO)':<15} {m1:<20.4f} {m2:<20.4f} {diff:<15.4f}\n")
        print(f"Comparison saved to {comp_file}")

    return results1, results2

if __name__ == "__main__":
    # ==================== 配置参数 ====================
    root_dir = "/mnt/disk/lyf/datasets/"   # 数据集根目录
    folder_list = ["test2"]                     # 子文件夹列表
    pred1_dir = "pred_service1_test2"                     # 服务1预测结果目录（相对路径）
    pred1_format = "labelme"                        # 服务1预测格式
    pred2_dir = "pred_qwen27_test2"                     # 服务2预测结果目录
    pred2_format = "labelme"                        # 服务2预测格式
    gt_format = "yolo"                              # 真实标注格式（'yolo' 或 'labelme'）
    output_dir = "./evaluation_results"             # 评估结果保存目录
    only_with_preds = True                          # 是否仅评估有预测的图像

    evaluate_services(
        dataset_name="my_combined",
        folder_list=folder_list,
        root_dir=root_dir,
        pred1_dir=pred1_dir,
        pred1_format=pred1_format,
        pred2_dir=pred2_dir,
        pred2_format=pred2_format,
        gt_format=gt_format,
        output_dir=output_dir,
        only_with_preds=only_with_preds
    )
    print("Evaluation completed.")